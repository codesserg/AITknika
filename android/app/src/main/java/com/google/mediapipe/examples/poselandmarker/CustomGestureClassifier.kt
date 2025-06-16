package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import org.tensorflow.lite.Interpreter
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

class CustomGestureClassifier(context: Context) {

    companion object {
        private const val TAG = "CustomGestureClassifier"
        private const val MODEL_FILENAME = "gesture_classification_model.tflite"
        private const val LABELS_FILENAME = "gesture_labels.json"
        private const val SEQUENCE_LENGTH = 30

        // Definir el número exacto de características que necesita el modelo
        // En Python: 33 landmarks de pose * 4 valores + 21 landmarks de mano izq * 3 + 21 landmarks de mano der * 3
        private const val POSE_LANDMARKS = 33
        private const val HAND_LANDMARKS = 21
        private const val POSE_VALUES_PER_LANDMARK = 4  // x, y, z, visibility
        private const val HAND_VALUES_PER_LANDMARK = 3  // x, y, z
        private const val FEATURES_PER_FRAME = (POSE_LANDMARKS * POSE_VALUES_PER_LANDMARK) +
                (HAND_LANDMARKS * HAND_VALUES_PER_LANDMARK * 2)
    }

    private var tfliteInterpreter: Interpreter? = null
    private val landmarkSequence = mutableListOf<FloatArray>()
    private var gestureLabels = mutableListOf<String>()
    private val context: Context

    init {
        this.context = context
        cargarModelo()
        cargarEtiquetas()
    }

    private fun cargarModelo() {
        try {
            context.assets.open(MODEL_FILENAME).use { inputStream ->
                val modelSize = inputStream.available()
                val modelBuffer = ByteBuffer.allocateDirect(modelSize)
                modelBuffer.order(ByteOrder.nativeOrder())

                val buffer = ByteArray(8192) // Buffer más pequeño para lectura por lotes
                var bytesRead: Int
                while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                    modelBuffer.put(buffer, 0, bytesRead)
                }
                modelBuffer.rewind() // Reiniciar el buffer al principio

                val options = Interpreter.Options()
                tfliteInterpreter = Interpreter(modelBuffer, options)
                Log.i(TAG, "Modelo TFLite cargado exitosamente con tamaño: $modelSize")

                // Registro de información del modelo
                val inputTensor = tfliteInterpreter!!.getInputTensor(0)
                val outputTensor = tfliteInterpreter!!.getOutputTensor(0)
                Log.i(TAG, "Forma del tensor de entrada: ${inputTensor.shape().contentToString()}")
                Log.i(TAG, "Forma del tensor de salida: ${outputTensor.shape().contentToString()}")
            }
        } catch (e: IOException) {
            Log.e(TAG, "Error al cargar el modelo TFLite: ${e.message}", e)
        }
    }

    private fun cargarEtiquetas() {
        try {
            context.assets.open(LABELS_FILENAME).use { inputStream ->
                val size = inputStream.available()
                val buffer = ByteArray(size)
                inputStream.read(buffer)
                val jsonString = String(buffer, Charsets.UTF_8)

                try {
                    // Primero intenta cargar como array JSON
                    val labelsArray = JSONArray(jsonString)
                    for (i in 0 until labelsArray.length()) {
                        gestureLabels.add(labelsArray.getString(i))
                    }
                } catch (e: JSONException) {
                    // Si falla, intenta cargar como objeto JSON
                    try {
                        val jsonObject = JSONObject(jsonString)
                        val labelsArray = jsonObject.getJSONArray("labels") // Asume que hay una clave "labels"
                        for (i in 0 until labelsArray.length()) {
                            gestureLabels.add(labelsArray.getString(i))
                        }
                    } catch (e2: JSONException) {
                        // Último intento: obtener las claves del objeto JSON
                        val jsonObject = JSONObject(jsonString)
                        val keysArray = jsonObject.names()
                        if (keysArray != null) {
                            for (i in 0 until keysArray.length()) {
                                gestureLabels.add(keysArray.getString(i))
                            }
                        }
                    }
                }

                Log.i(TAG, "Etiquetas de gestos cargadas: $gestureLabels")
            }
        } catch (e: IOException) {
            Log.e(TAG, "Error al cargar etiquetas: ${e.message}", e)
        } catch (e: JSONException) {
            Log.e(TAG, "Error al parsear JSON de etiquetas: ${e.message}", e)
        }
    }

    fun reconocerGesto(result: PoseLandmarkerResult): String {
        if (tfliteInterpreter == null) {
            Log.e(TAG, "El intérprete de TFLite es nulo")
            return "desconocido"
        }

        if (gestureLabels.isEmpty()) {
            Log.e(TAG, "Las etiquetas de gestos no están cargadas")
            return "desconocido"
        }

        if (result.landmarks().isEmpty()) {
            Log.w(TAG, "No se detectaron landmarks")
            return "sin_gesto"
        }

        return try {
            // Extraer el conjunto completo de landmarks simulando el mismo formato que en Python
            val landmarksFrameActual = extraerLandmarksCompletos(result)
            landmarkSequence.add(landmarksFrameActual)

            if (landmarkSequence.size >= SEQUENCE_LENGTH) {
                // Obtener secuencia reciente
                val secuenciaReciente = landmarkSequence.takeLast(SEQUENCE_LENGTH)

                // Verificar que tenemos el tamaño correcto de landmarks
                val tamañoEsperado = FEATURES_PER_FRAME
                if (secuenciaReciente[0].size != tamañoEsperado) {
                    Log.e(TAG, "Tamaño incorrecto de landmarks: ${secuenciaReciente[0].size}, se esperaba: $tamañoEsperado")
                    return "error_formato"
                }

                // Preparar datos para el modelo
                val inputData = Array(1) { Array(SEQUENCE_LENGTH) { FloatArray(tamañoEsperado) } }
                for (i in 0 until SEQUENCE_LENGTH) {
                    System.arraycopy(secuenciaReciente[i], 0, inputData[0][i], 0, tamañoEsperado)
                }

                // Convertir a ByteBuffer como espera TFLite
                val inputBuffer = ByteBuffer.allocateDirect(1 * SEQUENCE_LENGTH * tamañoEsperado * 4)
                inputBuffer.order(ByteOrder.nativeOrder())

                for (i in 0 until SEQUENCE_LENGTH) {
                    for (j in 0 until tamañoEsperado) {
                        inputBuffer.putFloat(inputData[0][i][j])
                    }
                }
                inputBuffer.rewind()

                // Preparar buffer de salida
                val outputProbability = Array(1) { FloatArray(gestureLabels.size) }

                // Ejecutar la inferencia
                tfliteInterpreter!!.run(inputBuffer, outputProbability)

                // Encontrar el índice con mayor probabilidad
                val indicePredecido = argmax(outputProbability[0])
                val confianza = outputProbability[0][indicePredecido]

                // Limpiar la secuencia si es demasiado larga
                if (landmarkSequence.size > SEQUENCE_LENGTH * 2) {
                    landmarkSequence.clear()
                    landmarkSequence.addAll(landmarkSequence.takeLast(SEQUENCE_LENGTH))
                }

                // Devolver predicción si la confianza es suficiente
                if (confianza >= 0.6f) {
                    val gestoPredecido = gestureLabels[indicePredecido]
                    Log.i(TAG, "Predicción: $gestoPredecido con confianza: $confianza")
                    gestoPredecido
                } else {
                    Log.i(TAG, "Predicción con baja confianza: $confianza")
                    "sin_gesto"
                }
            } else {
                Log.d(TAG, "Construyendo secuencia: ${landmarkSequence.size}/$SEQUENCE_LENGTH")
                "recopilando_datos"
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error durante el reconocimiento de gestos: ${e.message}", e)
            "error"
        }
    }

    /**
     * Extrae un conjunto completo de landmarks simulando el mismo formato del código Python
     * El código Python usaba: pose (33 puntos x 4 valores) + mano izq (21 puntos x 3 valores) + mano der (21 puntos x 3 valores)
     */
    private fun extraerLandmarksCompletos(result: PoseLandmarkerResult): FloatArray {
        val landmarks = FloatArray(FEATURES_PER_FRAME)

        // Inicializar todos los valores a cero
        landmarks.fill(0f)

        // Añadir puntos de pose disponibles
        if (result.landmarks().isNotEmpty()) {
            val poseLandmarks = result.landmarks()[0]

            // Copiar landmarks de pose disponibles
            for (i in poseLandmarks.indices) {
                if (i < POSE_LANDMARKS) {
                    val baseIndex = i * POSE_VALUES_PER_LANDMARK
                    landmarks[baseIndex] = poseLandmarks[i].x()
                    landmarks[baseIndex + 1] = poseLandmarks[i].y()
                    landmarks[baseIndex + 2] = poseLandmarks[i].z()
                    //landmarks[baseIndex + 3] = poseLandmarks[i].visibility()
                }
            }
        }

        // Nota: Los puntos de las manos se quedan como ceros porque no tenemos esa información
        // en el resultado de PoseLandmarkerResult.
        // El inicio de estos landmarks sería:
        // - Mano izquierda: POSE_LANDMARKS * POSE_VALUES_PER_LANDMARK
        // - Mano derecha: POSE_LANDMARKS * POSE_VALUES_PER_LANDMARK + HAND_LANDMARKS * HAND_VALUES_PER_LANDMARK

        return landmarks
    }

    private fun argmax(array: FloatArray): Int {
        var indiceMax = 0
        var valorMax = array[0]
        for (i in 1 until array.size) {
            if (array[i] > valorMax) {
                valorMax = array[i]
                indiceMax = i
            }
        }
        return indiceMax
    }

    // Método para liberar recursos cuando el clasificador ya no se necesita
    fun cerrar() {
        tfliteInterpreter?.close()
        tfliteInterpreter = null
        landmarkSequence.clear()
    }
}