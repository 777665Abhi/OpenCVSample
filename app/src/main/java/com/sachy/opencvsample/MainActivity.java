package com.sachy.opencvsample;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC1;

public class MainActivity extends AppCompatActivity implements View.OnClickListener, AdapterView.OnItemSelectedListener {
    TextView tvPickImage, tvAnalysis, tvResult;
    ImageView ivGalleryImage, ivResultImage;
    Spinner options;
    public static final int PICK_IMAGE = 1;
    Mat unknown, image, marker, closing, opening, output, imagemask, singleGrainMat, grainMat, finalMat, finalGrainMat;
    int position;
    int constantPos = 40;
    ArrayList<Bitmap> allGrainsList = new ArrayList<>();
    Bitmap galleryBitmap;
    Mat src;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Init
        tvPickImage = findViewById(R.id.tvPickImage);
        tvAnalysis = findViewById(R.id.tvAnalysis);
        tvResult = findViewById(R.id.tvResult);
        ivGalleryImage = findViewById(R.id.ivGalleryImage);
        ivResultImage = findViewById(R.id.ivResultImage);
        options = findViewById(R.id.options);
        tvPickImage.setOnClickListener(this);
        tvAnalysis.setOnClickListener(this);
        options.setOnItemSelectedListener(this);
        OpenCVLoader.initDebug();
    }


    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.tvPickImage:
                intentForGallery();
                break;

            case R.id.tvAnalysis:
                ivResultImage.setImageBitmap(genImgFromMat(findContour(galleryBitmap)));
                break;
        }
    }

    public Mat findContour(Bitmap bitmap) {
        src = new Mat(bitmap.getWidth(), bitmap.getHeight(), CV_8UC1);
        Utils.bitmapToMat(bitmap, image);
        //Converting the source image to binary
        Mat gray = new Mat(src.rows(), src.cols(), src.type());
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
        Mat binary = new Mat(src.rows(), src.cols(), src.type(), new Scalar(0));
        Imgproc.threshold(gray, binary, 100, 255, Imgproc.THRESH_BINARY_INV);
        //Finding Contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchey = new Mat();
        Imgproc.findContours(binary, contours, hierarchey, Imgproc.RETR_TREE,
                Imgproc.CHAIN_APPROX_SIMPLE);
        //Drawing the Contours
        Scalar color = new Scalar(0, 0, 255);
        Imgproc.drawContours(src, contours, -1, color, 2, Imgproc.LINE_8,
                hierarchey, 2, new Point());
        return src;
    }

    public Bitmap genImgFromMat(Mat src) {
        final Bitmap bitmap =
                Bitmap.createBitmap(src.width(), src.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(src, bitmap);
        return bitmap;
    }


    public void processImage(Bitmap bitmap) {
        //1
        image = new Mat(bitmap.getWidth(), bitmap.getHeight(), CV_8UC1);
        Utils.bitmapToMat(bitmap, image);
        Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2RGB);
        //2
        Mat hsv = new Mat();
        Imgproc.cvtColor(image, hsv, Imgproc.COLOR_RGB2HSV);

        int h_min = 76, h_max = 127, s_min = 81, s_max = 255, v_min = 0, v_max = 255;

//        int[] lower_b = {h_min, s_min, v_min};
//        int[] upper_b = {h_max, s_max, v_max};
        Scalar lower_b = new Scalar(h_min, s_min, v_min);
        Scalar upper_b = new Scalar(h_max, s_max, v_max);
        Core.inRange(hsv, lower_b, upper_b, hsv);

        //3
        Mat thresh_mask = new Mat();
        Mat full_mask = new Mat(hsv.size(), CV_8U, new Scalar(255));
        Core.subtract(full_mask, hsv, thresh_mask);

        //4
        Mat edges = new Mat();
        Imgproc.Canny(image, edges, 80, 100);

        //5
        unknown = Mat.zeros(edges.size(), CV_8U);

        for (int i = 0; i < edges.rows(); i++) {
            for (int j = 0; j < edges.cols(); j++) {
                double[] data = edges.get(i, j); //Stores element in an array
                //If want to Change the pixel
                for (int k = 0; k < edges.channels(); k++) //Runs for the available number of channels
                {
                    if (data[k] >= 0.2)
                        data[k] = 255.0; //Pixel modification done here
                }
                //If want to Change the pixel
                edges.put(i, j, data); //Puts element back into matrix
            }
        }
        unknown = edges;

        //6
        Mat thresh = new Mat();
        Core.subtract(thresh_mask, unknown, thresh);
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {

    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

    }

    private void intentForGallery() {
        Intent pickPhoto = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(pickPhoto, PICK_IMAGE, null);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE && resultCode == RESULT_OK) {

            final Bundle extras = data.getExtras();
            Uri imageUri = data.getData();
            if (imageUri != null) {
//            ivGalleryImage.setImageURI(imageUri);
                try {
                    Bitmap bitmap = bitmapFromUri(imageUri);
                    ivGalleryImage.setImageBitmap(bitmap);
                    galleryBitmap = bitmap;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public Bitmap bitmapFromUri(Uri imageUri) throws IOException {
        Bitmap bitmap;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(), imageUri));
        } else {
            bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
        }
        return bitmap;
    }
}