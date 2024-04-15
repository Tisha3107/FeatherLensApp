
package org.tensorflow.lite.examples.soundclassifier

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.webkit.WebSettings
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import org.tensorflow.lite.examples.soundclassifier.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

  private lateinit var soundClassifier: SoundClassifier
  private lateinit var binding: ActivityMainBinding

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    binding = ActivityMainBinding.inflate(layoutInflater)
    setContentView(binding.root)

    val windowMetrics = windowManager.currentWindowMetrics
    val width = windowMetrics.bounds.width()
    val paramsWebview: ViewGroup.LayoutParams = binding.webview.getLayoutParams() as ViewGroup.LayoutParams
    paramsWebview.height = (width / 1.8f).toInt()
    val paramsIcon: ViewGroup.LayoutParams = binding.icon.getLayoutParams() as ViewGroup.LayoutParams
    paramsIcon.height = (width / 1.8f).toInt()

    soundClassifier = SoundClassifier(this, binding, SoundClassifier.Options()).also {
      it.lifecycleOwner = this
    }
    binding.gps.setText(getString(R.string.latitude)+": --.-- / " + getString(R.string.longitude) + ": --.--" )
    binding.webview.setWebViewClient(object : MlWebViewClient(this) {})
    binding.webview.settings.setDomStorageEnabled(true)
    binding.webview.settings.setJavaScriptEnabled(true)

    binding.fab.setOnClickListener {
      if (binding.progressHorizontal.isIndeterminate) binding.progressHorizontal.setIndeterminate(false)
      else binding.progressHorizontal.setIndeterminate(true)
    }
    binding.bottomNavigationView.setOnNavigationItemSelectedListener { item ->
      when (item.itemId) {
        R.id.action_view -> {
          intent = Intent(this, ViewActivity::class.java)
          startActivity(intent)
        }
        R.id.action_share -> {
          val database = BirdDBHelper.getInstance(this)
          val intent = Intent(Intent.ACTION_SEND)
          val shareBody = database.exportAllEntriesAsCSV().joinToString("\n")
          intent.setType("text/plain")
          intent.putExtra(Intent.EXTRA_TEXT, shareBody)
          startActivity(Intent.createChooser(intent, ""))
        }
        R.id.action_delete -> {
          val database = BirdDBHelper.getInstance(this)
          database.clearAllEntries()
          Toast.makeText(this, getString(R.string.clear_db),Toast.LENGTH_SHORT).show()
        }
      }
      true
    }

    if (GithubStar.shouldShowStarDialog(this)) GithubStar.starDialog(this, "https://github.com/woheller69/whoBIRD")

    requestPermissions()

  }

  override fun onResume() {
    super.onResume()
    Location.requestLocation(this, soundClassifier)
    if (!checkLocationPermission()){
      Toast.makeText(this, this.resources.getString(R.string.error_location_permission), Toast.LENGTH_SHORT).show()
    }
    if (checkMicrophonePermission()){
      soundClassifier.start()
    } else {
      Toast.makeText(this, this.resources.getString(R.string.error_audio_permission), Toast.LENGTH_SHORT).show()
    }
    keepScreenOn(true)
  }

  override fun onPause() {
    super.onPause()
    Location.stopLocation(this)
    if (soundClassifier.isRecording) soundClassifier.stop()
  }

  private fun checkMicrophonePermission(): Boolean {
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO ) == PackageManager.PERMISSION_GRANTED) {
      return true
    } else {
      return false
    }
  }

  private fun checkLocationPermission(): Boolean {
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
      return true
    } else {
      return false
    }
  }

  private fun requestPermissions() {
    val perms = mutableListOf<String>()
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
      perms.add(Manifest.permission.RECORD_AUDIO)
    }
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
      perms.add(Manifest.permission.ACCESS_COARSE_LOCATION)
    }
    if (!perms.isEmpty()) requestPermissions(perms.toTypedArray(), REQUEST_PERMISSIONS)
  }

  private fun keepScreenOn(enable: Boolean) =
    if (enable) {
      window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    } else {
      window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

  companion object {
    const val REQUEST_PERMISSIONS = 1337
  }

  fun reload(view: View) {
    binding.webview.settings.setCacheMode(WebSettings.LOAD_DEFAULT)
    binding.webview.reload()
  }
}
