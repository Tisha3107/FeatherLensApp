package org.tensorflow.lite.examples.soundclassifier;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageButton;

public class HomePageActivity extends AppCompatActivity {

    ImageButton camera, mic;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home_page);
        camera=findViewById(R.id.camera);
        mic=findViewById(R.id.voice);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent mainActivity=new Intent(HomePageActivity.this, image.class);
                startActivity(mainActivity);
            }
        });

        mic.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent micActivity=new Intent(HomePageActivity.this, MainActivity.class);
                startActivity(micActivity);
            }
        });
    }
}