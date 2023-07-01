package com.example.face;

import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.Surface;
import android.view.TextureView;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
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
import java.util.Collections;
import java.util.List;

public class LiveVedio extends AppCompatActivity {
    CameraManager cameraManager;
    CameraDevice cameraDevice;
    Handler handler;
    TextureView textureView;

    ImageView imageView;
    Bitmap bitmap,bitmap1;

    Matrix matrix;
    static final String TAG = "FACE_DETECT_LIVE_TAG";
    FaceDetector detector;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_live_vedio);
        getPermission();
        HandlerThread handlerThread = new HandlerThread("videoThread");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
        textureView = findViewById(R.id.textureView);
        imageView = findViewById(R.id.imageView);

        FaceDetectorOptions realTimeFdo =
                new FaceDetectorOptions.Builder()
                        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                        .build();

        detector = FaceDetection.getClient(realTimeFdo);

        textureView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {

            @Override
            public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int width, int height) {
                openCamera();
            }

            @Override
            public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int width, int height) {
            }

            @Override
            public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
                return false;
            }

            @Override
            public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {
                bitmap = textureView.getBitmap();
                bitmap1 = ResizeImage(bitmap);
                PrepareImage(bitmap1);
            }
        });

    }

    //get les pertmission
    private void getPermission() {
        if (ContextCompat.checkSelfPermission(this, "android.permission.CAMERA") != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{"android.permission.CAMERA"}, 101);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults.length > 0 && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            getPermission();
        }
    }


    //open Camera
    @SuppressLint("MissingPermission")
    private void openCamera() {
        try {
            cameraManager = (CameraManager) getSystemService(CAMERA_SERVICE);

            cameraManager.openCamera(cameraManager.getCameraIdList()[1], new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice camera) {
                    cameraDevice = camera;
                    SurfaceTexture surfaceTexture = textureView.getSurfaceTexture();
                    surfaceTexture.setDefaultBufferSize(textureView.getWidth(), textureView.getHeight());
                    Surface surface = new Surface(surfaceTexture);

                    try {
                        final CameraCaptureSession.CaptureCallback captureCallback = new CameraCaptureSession.CaptureCallback() {
                        };

                        cameraDevice.createCaptureSession(Collections.singletonList(surface), new CameraCaptureSession.StateCallback() {
                            @Override
                            public void onConfigured(CameraCaptureSession session) {
                                try {
                                    CaptureRequest.Builder captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                                    captureRequestBuilder.addTarget(surface);

                                    session.setRepeatingRequest(captureRequestBuilder.build(), captureCallback, handler);
                                } catch (Exception e) {
                                    e.printStackTrace();
                                }
                            }

                            @Override
                            public void onConfigureFailed(CameraCaptureSession session) {
                            }
                        }, handler);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

                @Override
                public void onDisconnected(CameraDevice camera) {
                }

                @Override
                public void onError(CameraDevice camera, int error) {
                }
            }, handler);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }



    //traitement de image
    Bitmap ResizeImage(Bitmap bitmap){
        Bitmap bitmap1;
        int width = Math.max(bitmap.getWidth(), 480);
        int heigh = Math.max(bitmap.getHeight(), 360);
        bitmap1 = Bitmap.createScaledBitmap(bitmap, width, heigh, true);
        return bitmap1;
    }

    private void PrepareImage(Bitmap bitmap){
        Log.d(TAG,"Prepare Image");
        /////
        matrix = new Matrix();
        matrix.setScale(-1,1);
        /////

        Bitmap smallerBitmap= Bitmap.createBitmap(bitmap,0,0,bitmap.getWidth(),bitmap.getHeight(),matrix,true);

        InputImage inputImage = InputImage.fromBitmap(smallerBitmap,0);

        detector.process(inputImage)
                .addOnSuccessListener(faces -> {
                    Log.d(TAG,"onSuccess : Number of faces detected :"+faces.size());

                    if(faces.size()>=1) {
                        cropDetectedFaces(smallerBitmap, faces);

                    }

                })
                .addOnFailureListener(e -> {
                    //Detsection failed
                    Log.e(TAG,"onFailed: ",e);
                    Toast.makeText(LiveVedio.this,"Detection failed",Toast.LENGTH_SHORT).show();
                });
    }

    private void cropDetectedFaces(Bitmap bitmap, List<Face> faces) {
        Log.d(TAG,"crapDetectedFaces");
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
            Log.d(TAG, "cropBitmap: ");
            Bitmap caroppedBitmap = Bitmap.createBitmap(bitmap,
                    x,
                    y,
                    // (x * width > bitmap.getWidth()) ? bitmap.getWidth() - x : width,
                    // (y + height > bitmap.getHeight()) ? bitmap.getHeight() - y : height
                    width>=height ? height +10 : width,
                    width>=height ? height : width + 10
            );
            int val = preduction(caroppedBitmap);


            /////////////////////////////////////////////////
            // Assuming you have a Bitmap object named 'bitmap'



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

// Draw the rectangle on the Canvas
            canvas.drawRect(x, y, right, bottom, paint);

            j++;
        }
        mutableBitmap = Bitmap.createBitmap(mutableBitmap,0,0,bitmap.getWidth(),bitmap.getHeight(),matrix,true);
        imageView.setImageBitmap(mutableBitmap);
    }

    private int preduction(Bitmap bitmap) {
        Log.d(TAG, "preduction: ");
        int i1;
        try {
            FemalevsmaleMobilenetv2Ft80f1 model = FemalevsmaleMobilenetv2Ft80f1.newInstance(LiveVedio.this);
            Log.d(TAG, "modelf ");
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 160, 160, 3}, DataType.FLOAT32);
            int[] intValues = new int[160 * 160];
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
            i1 = sigmoidValue < 0.5 ? 0 : 1;

            return i1;
        } catch (IOException e) {

            return -1;
        }
        //return i1;
    }

}