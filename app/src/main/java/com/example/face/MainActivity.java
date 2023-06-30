package com.example.face;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.face.ml.FemalevsmaleMobilenetv2Ft80f1;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    Button selectBtn,predictBtn,captureBtn,liveBtn;
    ImageView imageView;
    TextView result;
    private Bitmap bitmap ;
    private static final String TAG = "FACE_DETECT_TAG";
    FaceDetector detector;
    private Bitmap bitmap1;

    private Matrix matrix;
    private final int SELECT_IMAGE = 10;
    private final int CAPTURE_IMAGE = 12;
    private final int REQUEST_PERMISSION_CAMERA = 20;
    private int witdhaff,heightaff;
    private FemalevsmaleMobilenetv2Ft80f1 model;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //INITIALIZATION DES VARIABLES
        selectBtn = findViewById(R.id.selectBtn);
        predictBtn = findViewById(R.id.predictBtn);
        captureBtn = findViewById(R.id.cptureBtn);
        imageView = findViewById(R.id.imageView);
        result = findViewById(R.id.result);
        liveBtn = findViewById(R.id.live);
        FaceDetectorOptions realTimeFdo =
                new FaceDetectorOptions.Builder()
                    .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                    .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                    .build();

        detector = FaceDetection.getClient(realTimeFdo);



        //Select image
        selectBtn.setOnClickListener(view -> {
            Intent intent = new Intent();
            intent.setAction(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            startActivityForResult(intent, SELECT_IMAGE);

        });

        //Preduction
        predictBtn.setOnClickListener(view -> {
            PrepareImage(bitmap1);
            changeBtnheigth(predictBtn,0);
        });

        //Capture image
        captureBtn.setOnClickListener(view -> {
            getPermission();

        });

        //live vedio
        liveBtn.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, LiveVedio.class);
            startActivity(intent);
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        detector.close();
        model.close();

    }

    //preparation d'image pour google face detection
    @SuppressLint("SetTextI18n")
    private void PrepareImage(Bitmap bitmap){
        Log.d(TAG,"Prepare Image");
        matrix = new Matrix();
        matrix.setScale(-1,1);
        Bitmap smallerBitmap= Bitmap.createBitmap(bitmap,0,0,bitmap.getWidth(),bitmap.getHeight(),matrix,true);

        InputImage inputImage = InputImage.fromBitmap(smallerBitmap,0);

        detector.process(inputImage)
            .addOnSuccessListener(faces -> {
                Log.d(TAG,"onSuccess : Number of faces detected :"+faces.size());

                if(faces.size()>=1) {
                    cropDetectedFaces(smallerBitmap, faces);
                    result.setText("");
                }
                else{
                    result.setText("No faces detected");
                    Toast.makeText(MainActivity.this,"No faces detected",Toast.LENGTH_SHORT).show();
                }
            })
            .addOnFailureListener(e -> {
                //Detsection failed
                Log.e(TAG,"onFailed: ",e);
                Toast.makeText(MainActivity.this,"Detection failed",Toast.LENGTH_SHORT).show();
            });
    }

    private void cropDetectedFaces(Bitmap bitmap, List<Face> faces) {
        Log.d(TAG,"cropDetectedFaces");
        int j = 0;
        // Create a mutable Bitmap from the original Bitmap
        Bitmap flip = Bitmap.createBitmap(bitmap1,0,0,bitmap.getWidth(),bitmap.getHeight(),matrix,true);
        Bitmap mutableBitmap = flip.copy(Bitmap.Config.ARGB_8888, true);


        // Create a Canvas object with the mutable Bitmap
        Canvas canvas = new Canvas(mutableBitmap);

        for (Face ignored : faces){
            Rect rect = faces.get(j).getBoundingBox();

            int x = Math.max(rect.left, 0);
            int y = Math.max(rect.top, 0);
            int width = rect.width();
            int height = rect.height();
            width = Math.min(width, bitmap.getWidth());
            height = Math.min(height, bitmap.getHeight());
            Log.d(TAG, "cropBitmap: ");
            Bitmap caroppedBitmap = Bitmap.createBitmap(bitmap,
                    x,
                    y,
                   // (x * width > bitmap.getWidth()) ? bitmap.getWidth() - x : width,
                   // (y + height > bitmap.getHeight()) ? bitmap.getHeight() - y : height
                    Math.min(width, height),
                    Math.min(width, height)
            );

            int val = preduction(caroppedBitmap);
            // Create a Paint object for drawing
            Paint paint = new Paint();
            if(val==1){
                paint.setColor(Color.BLUE);
            }
            if(val==0){
                paint.setColor(Color.parseColor("#FF00FF"));
            }
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(5); // Set the width of the rectangle's border

            // Define the rectangle coordinates (left, top, right, bottom)
            Log.d(TAG, "x+width ");
            int right = width>=height ? height+10+x:width+x;
            int bottom = width>=height ? height+y:width+10+y;
            right = Math.min(right, bitmap.getWidth());
            bottom = Math.min(bottom, bitmap.getHeight());

            // Draw the rectangle on the Canvas
            canvas.drawRect(x, y, right, bottom, paint);
            j++;
        }
        mutableBitmap = Bitmap.createBitmap(mutableBitmap,0,0,bitmap.getWidth(),bitmap.getHeight(),matrix,true);

        mutableBitmap = Bitmap.createScaledBitmap(mutableBitmap, witdhaff, heightaff, true);
        imageView.setImageBitmap(mutableBitmap);
    }

    //Preduction using our model
    private int preduction(Bitmap bitmap) {
        Log.d(TAG, "preduction: ");
        int i1;
        try {
            //load model
            model = FemalevsmaleMobilenetv2Ft80f1.newInstance(MainActivity.this);
            Log.d(TAG, "model loaded ");
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 160, 160, 3}, DataType.FLOAT32);
            int[] intValues = new int[160 * 160];
            //prepare image for our model
            bitmap = Bitmap.createScaledBitmap(bitmap, 160, 160, true);
            bitmap.getPixels(intValues, 0, 160, 0, 0, 160, 160);
            float[] floatValues = new float[160 * 160 * 3];
            Log.d(TAG, "preduction:1 ");
            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];
                floatValues[i * 3] = ((val >> 16) & 0xFF);
                floatValues[i * 3 + 1] = ((val >> 8) & 0xFF);
                floatValues[i * 3 + 2] = (val & 0xFF);
            }
            Log.d(TAG, "preduction:2 ");
            inputFeature0.loadArray(floatValues, new int[]{1, 160, 160, 3});

            FemalevsmaleMobilenetv2Ft80f1.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            Log.d(TAG, "preduction: 3");
            float prediction = outputFeature0.getFloatValue(0);
            float sigmoidValue = (float) (1.0f / (1.0f + Math.exp(-prediction)));
            // 0: female  ; 1: male
            i1 = sigmoidValue < 0.5 ? 0 : 1;

            return i1;
        } catch (IOException e) {
            Log.d(TAG, "preduction: " + e);
            return -1;
        }
    }

    //gere les activites
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

        if (requestCode == SELECT_IMAGE) {
            if(data!=null) {
                Uri uri = data.getData();
                try {

                    result.setText("");
                    try {
                        // Open an input stream for the content URI
                        InputStream inputStream = this.getContentResolver().openInputStream(uri);

                        // Create BitmapFactory.Options object and set inJustDecodeBounds to true to get the bitmap dimensions
                        BitmapFactory.Options options = new BitmapFactory.Options();
                        options.inJustDecodeBounds = true;
                        BitmapFactory.decodeStream(inputStream, null, options);

                        // Calculate the sample size based on the bitmap dimensions and your desired dimensions
                        int sampleSize = calculateSampleSize(options);

                        // Reset the input stream and set inJustDecodeBounds to false
                        inputStream = this.getContentResolver().openInputStream(uri);
                        options.inJustDecodeBounds = false;
                        options.inSampleSize = sampleSize;

                        // Decode the subsampled bitmap
                        bitmap = BitmapFactory.decodeStream(inputStream, null, options);

                        // Use the bitmap as needed
                        // ...

                        // Close the input stream
                        inputStream.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    if(bitmap != null){
                        changeBtnheigth(predictBtn,100);
                        Log.d(TAG, "bitmap resize ");

                        witdhaff = bitmap.getWidth() + 480;
                        heightaff = Math.max((bitmap.getHeight()*witdhaff)/bitmap.getWidth(),360);
                        //result.setText(""+bitmap.getHeight()+"/"+bitmap.getWidth());
                        bitmap1 = ResizeImage(bitmap);
                        imageView.setImageBitmap(bitmap1);
                    }

                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }
        else if(requestCode == CAPTURE_IMAGE){
            result.setText("");
            if(data != null) {
                bitmap = (Bitmap) data.getExtras().get("data");
                witdhaff = bitmap.getWidth() + 480;
                heightaff = Math.max((bitmap.getHeight()*witdhaff)/bitmap.getWidth(),360);
                changeBtnheigth(predictBtn,100);
                bitmap1 = ResizeImage(bitmap);
                imageView.setImageBitmap(bitmap1);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private int calculateSampleSize(BitmapFactory.Options options) {
        int originalWidth = options.outWidth;
        int originalHeight = options.outHeight;
        int sampleSize = 1;

        if (originalWidth > 480 || originalHeight > 360) {
            int widthRatio = Math.round((float) originalWidth / (float) 480);
            int heightRatio = Math.round((float) originalHeight / (float) 360);

            sampleSize = Math.min(widthRatio, heightRatio);
        }

        return sampleSize;
    }
    Bitmap ResizeImage(Bitmap bitmap){
        Bitmap bitmap1;
        witdhaff = bitmap.getWidth() + 480;
        heightaff = Math.max((bitmap.getHeight()*witdhaff)/bitmap.getWidth(),360);
        bitmap1 = Bitmap.createScaledBitmap(bitmap, witdhaff, heightaff, true);
        return bitmap1;
    }

    @SuppressLint("SuspiciousIndentation")
    void changeBtnheigth(Button button, int height){
        ViewGroup.LayoutParams layoutParams=button.getLayoutParams();
        layoutParams.height= ViewGroup.LayoutParams.WRAP_CONTENT;
        if(height==0)
            layoutParams.height =0;
        button.setLayoutParams(layoutParams);
    }

    // Permission Camera
    private void getPermission() {
        if (ContextCompat.checkSelfPermission(this, "android.permission.CAMERA") != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{"android.permission.CAMERA"}, REQUEST_PERMISSION_CAMERA);
        }
        else {
            Intent intent =new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(intent,CAPTURE_IMAGE);
        }
    }
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        getPermission();

    }
}