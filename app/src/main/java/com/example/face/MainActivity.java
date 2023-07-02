package com.example.face;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
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
import java.util.List;

/**
 * Cette classe représente l'activité principale de l'application Android.
 * Elle permet de sélectionner une image, de la traiter avec le détecteur de visages de Google ML Kit,
 * de recadrer les visages détectés et de les classifier en tant que masculin ou féminin à l'aide d'un modèle TensorFlow Lite.
 */
public class MainActivity extends AppCompatActivity {

    private Button selectBtn, predictBtn, captureBtn, liveBtn;
    private ImageView imageView;
    private TextView result;
    private static final String TAG = "FACE_DETECT_TAG";
    private FaceDetector detector;
    private Bitmap resizedbitmap;
    private Matrix matrix;
    private final int SELECT_IMAGE = 10;
    private final int CAPTURE_IMAGE = 12;
    private final int REQUEST_PERMISSION_CAMERA = 20;
    private FemalevsmaleMobilenetv2Ft80f1 model;
    /**
     * Méthode appelée lors de la création de l'activité.
     * Elle initialise les éléments de l'interface utilisateur et configure la détection des visages.
     *
     * @param savedInstanceState Les données enregistrées de l'activité précédente (si disponible)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialisation des variables
        selectBtn = findViewById(R.id.selectBtn);
        predictBtn = findViewById(R.id.predictBtn);
        captureBtn = findViewById(R.id.cptureBtn);
        imageView = findViewById(R.id.imageView);
        result = findViewById(R.id.result);
        liveBtn = findViewById(R.id.live);

        // Configuration des options du détecteur de visages
        FaceDetectorOptions realTimeFdo =
                new FaceDetectorOptions.Builder()
                        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                        .build();

        // Initialisation du détecteur de visages
        detector = FaceDetection.getClient(realTimeFdo);

        // Sélection d'une image
        selectBtn.setOnClickListener(view -> {
            Intent intent = new Intent();
            intent.setAction(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            startActivityForResult(intent, SELECT_IMAGE);
        });

        // Prédiction
        predictBtn.setOnClickListener(view -> {
            PrepareImage(resizedbitmap);
            changeBtnVisibilite(predictBtn, false);
        });

        // Capture d'image
        captureBtn.setOnClickListener(view -> getPermission());

        // Vidéo en direct
        liveBtn.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, LiveVedio.class);
            startActivity(intent);
        });
    }
    /**
     * Appelée lorsque l'activité est en cours de destruction. Libère les ressources et effectue les opérations de nettoyage.
     * Remplace l'implémentation par défaut de la méthode fournie par le framework Android.
     */
    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Fermeture du détecteur de visages et du modèle
        detector.close();
        model.close();
    }

    /**
     * Prépare l'image pour la détection de visages de Google ML Kit.
     * @param bitmap L'image sélectionnée
     */
    public void PrepareImage(Bitmap bitmap) {
        Log.d(TAG, "Prepare Image");
        matrix = new Matrix();
        matrix.setScale(-1, 1);
        Bitmap smallerBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        // Création de l'objet InputImage à partir du Bitmap redimensionné
        InputImage inputImage = InputImage.fromBitmap(smallerBitmap, 0);

        // Traitement de l'image avec le détecteur de visages de Google ML Kit
        detector.process(inputImage)
                .addOnSuccessListener(faces -> {
                    Log.d(TAG, "onSuccess : Number of faces detected :" + faces.size());

                    if (faces.size() >= 1) {
                        cropDetectedFaces(smallerBitmap, faces);
                        result.setText("");
                    } else {
                        result.setText("No faces detected");
                        Toast.makeText(MainActivity.this, "No faces detected", Toast.LENGTH_SHORT).show();
                    }
                })
                .addOnFailureListener(e -> {
                    // Échec de la détection
                    Log.e(TAG, "onFailed: ", e);
                    Toast.makeText(MainActivity.this, "Detection failed", Toast.LENGTH_SHORT).show();
                });
    }

    /**
     * Recadre les visages détectés dans l'image.
     * @param bitmap L'image d'origine
     * @param faces La liste des visages détectés
     */
    public void cropDetectedFaces(Bitmap bitmap, List<Face> faces) {
        Log.d(TAG, "cropDetectedFaces");
        int j = 0;
        // Création d'un Bitmap mutable à partir du Bitmap d'origine
        Bitmap flip = Bitmap.createBitmap(resizedbitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        Bitmap mutableBitmap = flip.copy(Bitmap.Config.ARGB_8888, true);

        // Création d'un objet Canvas avec le Bitmap mutable
        Canvas canvas = new Canvas(mutableBitmap);

        for (Face ignored : faces) {
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
                    Math.min(width, height),
                    Math.min(width, height)
            );

            int val = prediction(caroppedBitmap);
            Paint paint = new Paint();
            if (val == 1) {
                paint.setColor(Color.BLUE);
            }
            if (val == 0) {
                paint.setColor(Color.parseColor("#FF00FF"));
            }
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(5);

            int right = width >= height ? height + 10 + x : width + x;
            int bottom = width >= height ? height + y : width + 10 + y;
            right = Math.min(right, bitmap.getWidth());
            bottom = Math.min(bottom, bitmap.getHeight());

            // Dessine le rectangle sur le Canvas
            canvas.drawRect(x, y, right, bottom, paint);
            j++;
        }
        mutableBitmap = Bitmap.createBitmap(mutableBitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        mutableBitmap = ResizeImage(mutableBitmap);
        imageView.setImageBitmap(mutableBitmap);
    }

    /**
     * Effectue la prédiction en utilisant le modèle TensorFlow Lite.
     * @param bitmap L'image à classifier
     * @return La classe prédite (0 pour féminin, 1 pour masculin)
     */
    public int prediction(Bitmap bitmap) {
        Log.d(TAG, "prediction: ");
        int i1;
        try {
            // Chargement du modèle
            model = FemalevsmaleMobilenetv2Ft80f1.newInstance(MainActivity.this);
            Log.d(TAG, "model loaded ");
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 160, 160, 3}, DataType.FLOAT32);
            int[] intValues = new int[160 * 160];

            // Préparation de l'image pour le modèle
            bitmap = Bitmap.createScaledBitmap(bitmap, 160, 160, true);
            bitmap.getPixels(intValues, 0, 160, 0, 0, 160, 160);
            float[] floatValues = new float[160 * 160 * 3];
            Log.d(TAG, "prediction:1 ");
            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];
                floatValues[i * 3] = ((val >> 16) & 0xFF);
                floatValues[i * 3 + 1] = ((val >> 8) & 0xFF);
                floatValues[i * 3 + 2] = (val & 0xFF);
            }
            Log.d(TAG, "prediction:2 ");
            inputFeature0.loadArray(floatValues, new int[]{1, 160, 160, 3});

            FemalevsmaleMobilenetv2Ft80f1.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            Log.d(TAG, "prediction: 3");
            float prediction = outputFeature0.getFloatValue(0);
            float sigmoidValue = (float) (1.0f / (1.0f + Math.exp(-prediction)));
            // 0: féminin  ; 1: masculin
            i1 = sigmoidValue < 0.5 ? 0 : 1;

            return i1;
        } catch (IOException e) {
            Log.d(TAG, "prediction: " + e);
            return -1;
        }
    }

    /**
     * Gère les activités et les résultats de retour.
     * @param requestCode Le code de la requête
     * @param resultCode Le code de résultat
     * @param data Les données de l'Intent
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

        Bitmap bitmap;
        if (requestCode == SELECT_IMAGE) {
            if (data != null) {
                Uri uri = data.getData();
                try {
                    result.setText("");
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    Log.d(TAG, "bitmap resize ");
                    int width = Math.max(bitmap.getWidth(), 480);
                    int heigh = Math.max(bitmap.getHeight(), 360);
                    resizedbitmap = Bitmap.createScaledBitmap(bitmap, width, heigh, true);
                    changeBtnVisibilite(predictBtn, true);
                    imageView.setImageBitmap(resizedbitmap);


                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        } else if (requestCode == CAPTURE_IMAGE) {
            result.setText("");
            if (data != null) {
                bitmap = (Bitmap) data.getExtras().get("data");
                changeBtnVisibilite(predictBtn, true);
                int width = Math.max(bitmap.getWidth(), 480);
                int heigh = Math.max(bitmap.getHeight(), 360);
                resizedbitmap = Bitmap.createScaledBitmap(bitmap, width, heigh, true);
                imageView.setImageBitmap(resizedbitmap);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    /**
     * Redimensionne l'image à afficher dans l'ImageView.
     * @param bitmap L'image d'origine
     * @return Le bitmap redimensionné
     */
    public Bitmap ResizeImage(Bitmap bitmap) {
        Bitmap bitmap1;
        int witdhaff = bitmap.getWidth() + 480;
        int heightaff = Math.max((bitmap.getHeight() * witdhaff) / bitmap.getWidth(), 360);
        bitmap1 = Bitmap.createScaledBitmap(bitmap, witdhaff, heightaff, true);
        return bitmap1;
    }

    /**
     * Change la hauteur du bouton en fonction de la visibilité spécifiée.
     * @param button Le bouton à modifier
     * @param visible Indique si le bouton doit être visible (true) ou masqué (false)
     */
    public void changeBtnVisibilite(Button button, boolean visible){
        ViewGroup.LayoutParams layoutParams=button.getLayoutParams();
        layoutParams.height= ViewGroup.LayoutParams.WRAP_CONTENT;
        if(!visible)
            layoutParams.height =0;
        button.setLayoutParams(layoutParams);
    }

    /**
     * Vérifie les autorisations pour la capture d'image.
     */
    public void getPermission() {
        if (ContextCompat.checkSelfPermission(this, "android.permission.CAMERA") != PackageManager.PERMISSION_GRANTED) {
            int REQUEST_PERMISSION_CAMERA = 20;
            ActivityCompat.requestPermissions(this, new String[]{"android.permission.CAMERA"}, REQUEST_PERMISSION_CAMERA);
        }else {
            openCamera();
        }
    }

    /**
     * Méthode appelée lorsque la demande de permissions de la caméra a été traitée.
     * Elle vérifie si les permissions ont été accordées. Si ce n'est pas le cas, une nouvelle demande est affichée.
     *
     * @param requestCode  Le code de la demande de permissions
     * @param permissions  Les permissions demandées
     * @param grantResults Les résultats des permissions accordées ou refusées
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        getPermission();
    }


    /**
     * Ouvre l'appareil photo pour capturer une image.
     */
    public void openCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(cameraIntent, CAPTURE_IMAGE);
    }
}
