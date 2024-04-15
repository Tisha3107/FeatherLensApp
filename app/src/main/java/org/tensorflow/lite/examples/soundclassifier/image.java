package org.tensorflow.lite.examples.soundclassifier;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import com.github.dhaval2404.imagepicker.ImagePicker;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.examples.soundclassifier.ml.BirdDetection;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileDescriptor;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class image extends AppCompatActivity {

    TextView result, demoTxt, classified, clickHere;
    ImageView imageView, arrowImage;
    ImageButton picture;

    int imageSize=224;      //default image size
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image);

        result=findViewById(R.id.result);
        demoTxt=findViewById(R.id.demoText);
        classified=findViewById(R.id.classified);
        clickHere=findViewById(R.id.click_here);
        imageView=findViewById(R.id.imageView);
        arrowImage=findViewById(R.id.demoArrow);

        picture=findViewById(R.id.button);

        demoTxt.setVisibility(View.VISIBLE);
        clickHere.setVisibility(View.GONE);
        arrowImage.setVisibility(View.VISIBLE);
        classified.setVisibility(View.GONE);
        result.setVisibility(View.GONE);

        picture.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                //launch camera if we have permission
                if(checkSelfPermission(Manifest.permission.CAMERA)== PackageManager.PERMISSION_GRANTED){
                    // Intent intent=new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    // startActivityForResult(intent, 1);
                    ImagePicker.with(image.this)
                            .crop()
                            .maxResultSize(1080,1080)
                            .start();
                }else{
                    //request camera permission
                    requestPermissions(new String[]{Manifest.permission.CAMERA},100);
                }
            }
        });

    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(resultCode==RESULT_OK){
            Uri imageUri = data.getData();
            ParcelFileDescriptor parcelFileDescriptor = null;
            try {
                parcelFileDescriptor = getContentResolver().openFileDescriptor(imageUri, "r");
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
            Bitmap image = BitmapFactory.decodeFileDescriptor(fileDescriptor);

            try {
                parcelFileDescriptor.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            int dimension=Math.min(image.getWidth(),image.getHeight());
            image= ThumbnailUtils.extractThumbnail(image,dimension, dimension);
            imageView.setImageBitmap(image);                                    //setting the image on the imageView

            demoTxt.setVisibility(View.GONE);
            clickHere.setVisibility(View.VISIBLE);
            arrowImage.setVisibility(View.GONE);
            classified.setVisibility(View.VISIBLE);
            result.setVisibility(View.VISIBLE);

            image=Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            classifyImage(image);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }




    private void classifyImage(Bitmap image) {
        try{
            BirdDetection model = BirdDetection.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer=ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            //get 1D array of 224*224 pixels in image
            int[] intValue=new int[imageSize * imageSize];
            image.getPixels(intValue,0, image.getWidth(),0,0, image.getWidth(), image.getHeight());

            //iterate over pixels and extract R,B,G values, add to bytebuffer
            int pixel=0;
            for ( int i=0; i<imageSize; i++){
                for( int j=0; j<imageSize; j++){
                    int val=intValue[pixel++]; //RBG
                    byteBuffer.putFloat(((val>>16) & 0xFF) * (1.f/255.f));
                    byteBuffer.putFloat(((val>>8) & 0xFF) * (1.f/255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f/255.f));

                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Run model inference and gets result.
            BirdDetection.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidence=outputFeature0.getFloatArray();

            //find the index of the class with the biggest confidence
            int maxPos=0;
            float maxConfidence=0;
            for (int i=0; i<confidence.length; i++){
                if(confidence[i]>maxConfidence){
                    maxConfidence=confidence[i];
                    maxPos = i;
                }
            }
            String[] classes={"AFRICAN EMERALD CUCKOO","ALBATROSS","AMERICAN FLAMINGO","ANNAS HUMMINGBIRD","BALD EAGLE","BIRD OF PARADISE","BLACK SWAN","BLUE HERON","BROWN HEADED COWBIRD","CALIFORNIA QUAIL","CHESTNUT WINGED CUCKOO","CHIPPING SPARROW","COCKATOO","COMMON STARLING","CROW","DARK EYED JUNCO","DUSKY LORY","EASTERN BLUEBIRD","EASTERN YELLOW ROBIN","EMPEROR PENGUIN","FAIRY BLUEBIRD","FRIGATE","GOLDEN EAGLE","GO AWAY BIRD","HORNED LARK","HOUSE SPARROW","INDIAN BUSTARD","IVORY GULL","JAPANESE ROBIN","LAUGHING GULL","KILLDEAR","MANGROVE CUCKOO","NICOBAR PIGEON","OSTRICH","ORANGE BRESTED BUNTING","PINK ROBIN","PEACOCK","ROSY FACED LOVEBIRD","ROYAL FLYCATCHER","SNOW GOOSE","SCARLET MACAW","TAILORBIRD","UMBRELLA BIRD","VULTURINE GUINEAFOWL","WILD TURKEY","WHITE BROWED CRAKE","YELLOW CACIQUE","YELLOW BELLIED FLOWERPECKER","WRENTIT","ZEBRA DOVE"};
            result.setText(classes[maxPos]);

            //to search the soil on Internet and validating it.
            result.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse("https://www.google.com/search?q=" + result.getText())));
                }
            });


            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
}
