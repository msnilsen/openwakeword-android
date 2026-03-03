plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("maven-publish")
}

android {
    namespace = "com.openwakeword"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        consumerProguardFiles("consumer-rules.pro")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    api("com.microsoft.onnxruntime:onnxruntime-android:1.17.1")
}

afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("release") {
                from(components["release"])
                groupId = "com.github.msnilsen"
                artifactId = "openwakeword-android"
                version = findProperty("VERSION_NAME")?.toString() ?: "0.1.0"

                pom {
                    name.set("openWakeWord Android")
                    description.set("Kotlin/Android port of openWakeWord — on-device wake word detection using ONNX Runtime")
                    url.set("https://github.com/msnilsen/openwakeword-android")
                    licenses {
                        license {
                            name.set("Apache License 2.0")
                            url.set("https://www.apache.org/licenses/LICENSE-2.0")
                        }
                    }
                }
            }
        }
    }
}
