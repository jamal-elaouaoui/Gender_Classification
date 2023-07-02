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

/**
 * Cette classe représente une activité permettant la détection en direct des visages à partir de la caméra.
 * Elle utilise la bibliothèque ML Kit de Google pour la détection des visages, recadrer les visages détectés
 * et de les classifier en tant que masculin ou féminin à l'aide d'un modèle TensorFlow Lite.
 */
public class LiveVedio extends AppCompatActivity {
    // Gestion de la caméra
    private CameraManager cameraManager;
    private CameraDevice cameraDevice;
    private Handler handler;
    private TextureView textureView;

    // Affichage des résultats
    private ImageView imageView;
    private Bitmap bitmap, resizedbitmap;

    private Matrix matrix;
    private static final String TAG = "FACE_DETECT_LIVE_TAG";
    private FaceDetector detector;
    private FemalevsmaleMobilenetv2Ft80f1 model;

    /**
     * Méthode appelée lors de la création de l'activité.
     * Elle initialise les éléments de l'interface utilisateur et configure la détection des visages.
     *
     * @param savedInstanceState Les données enregistrées de l'activité précédente (si disponible)
     */
    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_live_vedio);
        getPermission();
        // Initialisation des variables
        HandlerThread handlerThread = new HandlerThread("videoThread");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
        textureView = findViewById(R.id.textureView);
        imageView = findViewById(R.id.imageView);

        // Configuration des options du détecteur de visages
        FaceDetectorOptions realTimeFdo =
                new FaceDetectorOptions.Builder()
                        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                        .build();

        // Initialisation du détecteur de visages
        detector = FaceDetection.getClient(realTimeFdo);

        /**
         * Méthode pour configurer un TextureView et ajouter un SurfaceTextureListener.
         * Lorsque le SurfaceTexture est disponible, la caméra est ouverte.
         * Lorsque le SurfaceTexture est mis à jour, l'image est extraite du TextureView, redimensionnée, préparée et analysée.
         */
        textureView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {

            /**
             * Méthode appelée lorsque le SurfaceTexture est disponible pour l'affichage.
             * Elle ouvre la caméra pour la capture du flux vidéo.
             *
             * @param surfaceTexture Le SurfaceTexture utilisé pour l'affichage
             * @param width          La largeur du SurfaceTexture
             * @param height         La hauteur du SurfaceTexture
             */
            @Override
            public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int width, int height) {
                openCamera();
            }

            /**
             * Méthode appelée lorsque la taille du SurfaceTexture est modifiée.
             * Elle ne fait rien dans ce contexte.
             *
             * @param surfaceTexture Le SurfaceTexture
             * @param width          La nouvelle largeur du SurfaceTexture
             * @param height         La nouvelle hauteur du SurfaceTexture
             */
            @Override
            public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int width, int height) {
                // Ne fait rien
            }

            /**
             * Méthode appelée lorsque le SurfaceTexture est détruit.
             * Elle renvoie false pour indiquer que le SurfaceTexture n'a pas été détruit manuellement.
             *
             * @param surfaceTexture Le SurfaceTexture
             * @return false pour indiquer que le SurfaceTexture n'a pas été détruit manuellement
             */
            @Override
            public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
                return false;
            }

            /**
             * Méthode appelée lorsque le SurfaceTexture est mis à jour.
             * Elle extrait l'image du TextureView, la redimensionne, la prépare et l'analyse.
             *
             * @param surfaceTexture Le SurfaceTexture
             */
            @Override
            public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {
                bitmap = textureView.getBitmap();
                resizedbitmap = ResizeImage(bitmap);
                PrepareImage(resizedbitmap);
            }
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
     * Méthode privée pour obtenir les permissions de la caméra.
     * Si les permissions ne sont pas accordées, une demande de permission est affichée.
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
        //if (grantResults.length > 0 && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
        getPermission();
        //}
    }

    /**
     * Méthode pour ouvrir la caméra et configurer la capture du flux vidéo.
     * Elle utilise le CameraManager pour ouvrir la caméra spécifiée.
     * Lorsque la caméra est ouverte avec succès, un SurfaceTexture est obtenu à partir du TextureView,
     * un Surface est créé à partir du SurfaceTexture, et une session de capture est créée pour la caméra.
     * Enfin, une requête de capture est configurée et répétée en utilisant la session de capture.
     * Des rappels sont définis pour les états de la caméra et de la session de capture.
     */
    @SuppressLint("MissingPermission")
    public void openCamera() {
        try {
            cameraManager = (CameraManager) getSystemService(CAMERA_SERVICE);

            cameraManager.openCamera(cameraManager.getCameraIdList()[1], new CameraDevice.StateCallback() {

                /**
                 * Méthode appelée lorsque la caméra est ouverte avec succès.
                 * Elle initialise la variable cameraDevice avec la caméra ouverte.
                 * Un SurfaceTexture est obtenu à partir du TextureView et un Surface est créé à partir du SurfaceTexture.
                 * Ensuite, une session de capture est créée pour la caméra en utilisant la liste de surfaces, qui contient
                 * un seul élément : le Surface.
                 * Une requête de capture est configurée et répétée en utilisant la session de capture, le Surface cible
                 * et un objet de rappel CaptureCallback.
                 *
                 * @param camera La caméra ouverte
                 */
                @Override
                public void onOpened(CameraDevice camera) {
                    cameraDevice = camera;
                    SurfaceTexture surfaceTexture = textureView.getSurfaceTexture();
                    surfaceTexture.setDefaultBufferSize(textureView.getWidth(), textureView.getHeight());
                    Surface surface = new Surface(surfaceTexture);

                    try {
                        final CameraCaptureSession.CaptureCallback captureCallback = new CameraCaptureSession.CaptureCallback() {
                            // Callback de capture vide, sans implémentation supplémentaire
                        };

                        cameraDevice.createCaptureSession(Collections.singletonList(surface), new CameraCaptureSession.StateCallback() {

                            /**
                             * Méthode appelée lorsque la session de capture est configurée avec succès.
                             * Elle crée une requête de capture en utilisant le modèle TEMPLATE_PREVIEW et
                             * ajoute le Surface cible à la requête.
                             * Ensuite, la requête de capture est répétée en utilisant la session de capture,
                             * la requête de capture et un objet de rappel CaptureCallback.
                             *
                             * @param session La session de capture configurée
                             */
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

                            /**
                             * Méthode appelée lorsque la configuration de la session de capture a échoué.
                             *
                             * @param session La session de capture qui a échoué à se configurer
                             */
                            @Override
                            public void onConfigureFailed(CameraCaptureSession session) {
                                // Échec de la configuration de la session de capture
                            }
                        }, handler);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

                /**
                 * Méthode appelée lorsque la caméra est déconnectée.
                 *
                 * @param camera La caméra déconnectée
                 */
                @Override
                public void onDisconnected(CameraDevice camera) {
                    // Caméra déconnectée
                }

                /**
                 * Méthode appelée lorsqu'une erreur se produit avec la caméra.
                 *
                 * @param camera La caméra sur laquelle l'erreur s'est produite
                 * @param error  Le code d'erreur
                 */
                @Override
                public void onError(CameraDevice camera, int error) {
                    // Erreur de la caméra
                }
            }, handler);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }



    /**
     * Méthode privée pour redimensionner une image.
     *
     * @param bitmap L'image à redimensionner
     * @return L'image redimensionnée
     */
    public Bitmap ResizeImage(Bitmap bitmap){
        Bitmap bitmap1;
        int width = Math.max(bitmap.getWidth(), 480);
        int heigh = Math.max(bitmap.getHeight(), 360);
        bitmap1 = Bitmap.createScaledBitmap(bitmap, width, heigh, true);
        return bitmap1;
    }

    /**
     * Méthode pour préparer l'image pour la détection des visages.
     * Elle redimensionne et inverse horizontalement l'image d'entrée pour l'adapter à la détection des visages.
     * Ensuite, elle crée un objet InputImage à partir du Bitmap de l'image préparée.
     * L'objet Detector est utilisé pour traiter l'image d'entrée et détecter les visages.
     * Un rappel onSuccess est défini pour traiter les résultats de la détection des visages.
     * Si au moins un visage est détecté, la méthode cropDetectedFaces() est appelée pour recadrer les visages détectés.
     * Un rappel onFailure est défini pour gérer les erreurs de détection.
     *
     * @param bitmap L'image d'entrée
     */
    public void PrepareImage(Bitmap bitmap) {
        Log.d(TAG, "Prepare Image");

        // Inverser horizontalement l'image
        matrix = new Matrix();
        matrix.setScale(-1, 1);
        Bitmap smallerBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        // Créer un objet InputImage à partir du Bitmap
        InputImage inputImage = InputImage.fromBitmap(smallerBitmap, 0);

        // Traitement de l'image d'entrée avec le détecteur
        detector.process(inputImage)
                .addOnSuccessListener(faces -> {
                    Log.d(TAG, "onSuccess : Number of faces detected: " + faces.size());

                    if (faces.size() >= 1) {
                        // Recadrer les visages détectés
                        cropDetectedFaces(smallerBitmap, faces);
                    }
                })
                .addOnFailureListener(e -> {
                    // Échec de la détection
                    Log.e(TAG, "onFailed: ", e);
                    Toast.makeText(LiveVedio.this, "Detection failed", Toast.LENGTH_SHORT).show();
                });
    }

    /**
     * Méthode privée pour recadrer les visages détectés dans l'image.
     * Elle crée un bitmap mutable à partir de l'image, dessine des rectangles autour des visages détectés et affiche le résultat sur l'imageView.
     *
     * @param bitmap L'image d'origine
     * @param faces  La liste des visages détectés
     */
    public void cropDetectedFaces(Bitmap bitmap, List<Face> faces) {
        Log.d(TAG,"crapDetectedFaces");
        int j = 0;
        // Create a mutable Bitmap from the original Bitmap
        Bitmap flip = Bitmap.createBitmap(resizedbitmap,0,0,bitmap.getWidth(),bitmap.getHeight(),matrix,true);
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

                    Math.max(width, height),
                    Math.max(width, height)
            );
            int val = prediction(caroppedBitmap);
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

    /**
     * Méthode pour effectuer une prédiction sur une image donnée à l'aide d'un modèle de classification.
     * Elle utilise le modèle FemalevsmaleMobilenetv2Ft80f1 pour effectuer la prédiction.
     * L'image d'entrée est redimensionnée pour correspondre aux dimensions d'entrée du modèle.
     * Les valeurs des pixels de l'image sont normalisées et chargées dans un TensorBuffer en tant qu'entrée du modèle.
     * Le modèle est utilisé pour effectuer la prédiction et retourne une valeur de prédiction.
     * La valeur de prédiction est ensuite convertie en une valeur entre 0 et 1 à l'aide de la fonction sigmoïde.
     * Si la valeur de prédiction est inférieure à 0.5, la méthode retourne 0 pour représenter "female", sinon elle retourne 1 pour représenter "male".
     * En cas d'erreur lors de la création du modèle, la méthode retourne -1.
     *
     * @param bitmap L'image d'entrée
     * @return La prédiction : 0 pour "female", 1 pour "male", -1 en cas d'erreur
     */
    public int prediction(Bitmap bitmap) {
        Log.d(TAG, "preduction: ");
        int i1;
        try {
            // Charger le modèle
            model = FemalevsmaleMobilenetv2Ft80f1.newInstance(LiveVedio.this);
            Log.d(TAG, "model loaded");

            // Créer un TensorBuffer pour l'entrée du modèle
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 160, 160, 3}, DataType.FLOAT32);

            // Redimensionner l'image pour correspondre aux dimensions d'entrée du modèle
            bitmap = Bitmap.createScaledBitmap(bitmap, 160, 160, true);

            // Extraire les valeurs des pixels de l'image et les normaliser
            int[] intValues = new int[160 * 160];
            bitmap.getPixels(intValues, 0, 160, 0, 0, 160, 160);
            float[] floatValues = new float[160 * 160 * 3];
            Log.d(TAG, "preduction: 1");
            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];
                floatValues[i * 3] = ((val >> 16) & 0xFF);
                floatValues[i * 3 + 1] = ((val >> 8) & 0xFF);
                floatValues[i * 3 + 2] = (val & 0xFF);
            }
            Log.d(TAG, "preduction: 2");

            // Charger les valeurs normalisées dans le TensorBuffer d'entrée du modèle
            inputFeature0.loadArray(floatValues, new int[]{1, 160, 160, 3});

            // Utiliser le modèle pour effectuer la prédiction
            FemalevsmaleMobilenetv2Ft80f1.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            Log.d(TAG, "preduction: 3");
            float prediction = outputFeature0.getFloatValue(0);

            // Appliquer la fonction sigmoïde à la valeur de prédiction pour obtenir une valeur entre 0 et 1
            float sigmoidValue = (float) (1.0f / (1.0f + Math.exp(-prediction)));

            // Convertir la valeur de prédiction en 0 ou 1 en fonction du seuil de 0.5
            i1 = sigmoidValue < 0.5 ? 0 : 1;

            return i1;
        } catch (IOException e) {
            // En cas d'erreur lors de la création du modèle
            return -1;
        }
    }
}
