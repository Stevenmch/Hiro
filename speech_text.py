import json
import multiprocessing as mp
import random
import time
from typing import Text

import numpy as np
import soundfile as sf
import tflite_runtime.interpreter as tflite

import pyaudio
import wave

import pandas as pd
from datetime import datetime
from datetime import date 
from ds_ctcdecoder import Alphabet, Scorer, swigwrapper
from transformers import pipeline
import socket
import sys


checkpoint_file = "C:/Users/StevenMendezChipatec/Steven/Hiro/NLP/brain/models/model_quantized_Revision.tflite"
test_wav_path = "C:/Users/StevenMendezChipatec/Steven/Hiro/NLP/brain/data/output.wav"
alphabet_path = "C:/Users/StevenMendezChipatec/Steven/Hiro/NLP/brain/models/alphabet_ES.json"
ds_alphabet_path = "C:/Users/StevenMendezChipatec/Steven/Hiro/NLP/brain/models/alphabet_ES.txt"
ds_scorer_path = "C:/Users/StevenMendezChipatec/Steven/Hiro/NLP/brain/models/kenlm_es_n12.scorer"
beam_size = 1024
sample_rate = 16000

# Experiment a little with those values to optimize inference
chunk_size = int(1.0 * sample_rate)
frame_overlap = int(2.0 * sample_rate)
char_offset = 4

with open(alphabet_path, "r", encoding="utf-8") as file:
    alphabet = json.load(file)

ds_alphabet = Alphabet(ds_alphabet_path)
ds_scorer = Scorer(
    alpha=0.931289039105002,
    beta=1.1834137581510284,
    scorer_path=ds_scorer_path,
    alphabet=ds_alphabet,
)
ds_decoder = swigwrapper.DecoderState()
ds_decoder.init(
    alphabet=ds_alphabet,
    beam_size=beam_size,
    cutoff_prob=1.0,
    cutoff_top_n=512,
    ext_scorer=ds_scorer,
    hot_words=dict(),
)
# ================
def enviar_unity(mensaje):
    # Creando Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # La tupla con la direccion y el puerto
    server_address = ('localhost', 65535)
    # Encode el mensaje
    mensaje = mensaje.encode()
    try:

        print('Esto es lo que vamos a enviar', mensaje)
        # Enviar datos 
        sent = sock.sendto(mensaje, server_address)
        print(server_address)
        # Recibe un mensaje del servidor
        print('Esperando el mensaje del servidor...')
        data, server = sock.recvfrom(4096)
        print('Mensaje Recibido',data)

    finally:
        print('Cerrando el socket, ya no')
        

def fecha_dia(oracion):

    my_date = datetime.today()
    año = str(my_date.year)
    mes = str(my_date.month)
    dia = str(my_date.day)
    fecha = año+'-'+mes+'-'+dia
    dia_semana = my_date.weekday()
    # Dia de agendamiento, saber en numero que dia es el dia que queremos la tarea
    days = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo', 'hoy']

    for j in days:
        if 'hoy' in oracion:
            fecha_tarea2 = date(int(año),int(mes),int(dia))
            fecha_tarea2 = fecha_tarea2.strftime("%Y-%m-%d")
            return fecha_tarea2

        if j in oracion:
            contador =+ 1
            break

    dia_tarea = days.index(j)
    # Distancia entre los dias, para despues sumar esa distancia al calendario
    if dia_semana < dia_tarea:
        dist_dias = abs(dia_semana - dia_tarea)
    if dia_semana >= dia_tarea:
        dist_dias = 6-((dia_semana-dia_tarea)-1)
    # Funcion para saber cuantos dias tiene cada mes
    def es_bisiesto(year_actual: int) -> bool:
        return year_actual % 4 == 0 and (year_actual % 100 != 0 or year_actual % 400 == 0)

    def obtener_dias_del_mes(mes: int, year_actual: int) -> int:
        # Abril, junio, septiembre y noviembre tienen 30
        if mes in [4, 6, 9, 11]:
            return 30
        # Febrero depende de si es o no bisiesto
        if mes == 2:
            if es_bisiesto(year_actual):
                return 29
            else:
                return 28
        else:
        # En caso contrario, tiene 31 días
            return 31
    # Saber los dias del mes actual y hacer la suma para saber cuando es el dia que tengo que hacer la tarea, en este caso la tarea del viernes
    dias = obtener_dias_del_mes(int(mes), int(año))
    dias
    dia_hoy = my_date.day
    mes_hoy = my_date.month
    # Conocer dia de la tarea
    for i in range(0,dist_dias):
        if dia_hoy >= dias:
            # Cambio de mes, agregar mes + 1
            dia_hoy = 1
            mes_hoy = mes_hoy + 1
        else:
            dia_hoy = dia_hoy + 1
            
    # Fecha en string
    fecha_tarea = str(mes_hoy)+'-'+str(dia_hoy)+'-'+año
    # Fecha en formato date pero no como la queremos
    fecha_tarea = datetime.strptime(fecha_tarea,'%m-%d-%Y')
    # Aqui ya esta en formato fecha que queremos
    fecha_tarea2 = date(int(año),mes_hoy,dia_hoy)
    fecha_tarea2 = fecha_tarea2.strftime("%Y-%m-%d")
    return fecha_tarea2
# ==================================================================================================
def consultar_agendar(sentencia):
    # Instacia de la clase, donde se escoge el modelo en español
    classifier = pipeline("zero-shot-classification", 
                       model="Recognai/bert-base-spanish-wwm-cased-xnli")
    # Aplicar modelo, donde recibe la oracion, y las clases
    clasificacion = classifier(sentencia, candidate_labels=["consultar", "agendar"])
    
    if clasificacion['labels'][0] == 'consultar':
        consultar(sentencia)
        print('consultar')
    else:
        recognize_day(sentencia)
        print('Agendar')

def consultar(frase_consultar):
    fecha_tarea = fecha_dia(frase_consultar)
    read_file = pd.read_csv("C:/Users/StevenMendezChipatec/Steven/Hiro/NLP/brain/data/Tabla_Tareas.csv")
    tareas_req = [read_file.loc[fila, 'Frase'] for fila in range(0, len(read_file)) if read_file.loc[fila, 'Fecha_Tarea']== fecha_tarea]
    for i in tareas_req: print('- ',i)
    tareas_string = ",".join(tareas_req)
    tareas_string = ','+ tareas_string
    enviar_unity(tareas_string)
    return tareas_req

def recognize_day(frase):

    days = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo']
    contador = 0
    # No sirve de mucho despues de que se creo el dataframe
    df = pd.DataFrame(columns=['Tarea', 'Dia', 'Hora', 'Duracion', 'Fecha' 'Frase'])
    # Fecha actual
    my_date = datetime.now()

    for j in days:
        if j in frase:
            contador =+ 1
            break
        if 'hoy' in frase:
            fecha_tarea = date(my_date.year,my_date.month,my_date.day)
            return fecha_tarea

    if contador == 1:
        print('El dia es:',j)
    else:
        print('No hay dias en la frase')
    dia_tarea = days.index(j)

    
    day_list = date(my_date.year, my_date.month, my_date.day)
    day_list = day_list.strftime("%Y-%m-%d")
    # Distancia entre los dias, para despues sumar esa distancia al calendario
    dia_semana = my_date.weekday()
    if dia_semana < dia_tarea:
        dist_dias = abs(dia_semana - dia_tarea)
    if dia_semana >= dia_tarea:
        dist_dias = 6-((dia_semana-dia_tarea)-1)
    # Funcion para saber cuantos dias tiene cada mes
    def es_bisiesto(year_actual: int) -> bool:
        return year_actual % 4 == 0 and (year_actual % 100 != 0 or year_actual % 400 == 0)

    def obtener_dias_del_mes(mes: int, year_actual: int) -> int:
        # Abril, junio, septiembre y noviembre tienen 30
        if mes in [4, 6, 9, 11]:
            return 30
        # Febrero depende de si es o no bisiesto
        if mes == 2:
            if es_bisiesto(year_actual):
                return 29
            else:
                return 28
        else:
        # En caso contrario, tiene 31 días
            return 31
    # Saber los dias del mes actual y hacer la suma para saber cuando es el dia que tengo que hacer la tarea, en este caso la tarea del viernes
    año = str(my_date.year)
    mes = str(my_date.month)
    dias = obtener_dias_del_mes(int(mes), int(año))
    dias
    dia_hoy = my_date.day
    mes_hoy = my_date.month
    # Conocer dia de la tarea
    print(dist_dias)
    for i in range(0,dist_dias):
        if dia_hoy >= dias:
            # Cambio de mes, agregar mes + 1
            dia_hoy = 1
            mes_hoy = mes_hoy + 1
        else:
            dia_hoy = dia_hoy + 1
            print(dia_hoy)

    fecha_tarea = date(int(año),mes_hoy,dia_hoy)
    fecha_tarea = fecha_tarea.strftime("%Y-%m-%d")
    # Read
    read_file = pd.read_csv("C:/Users/StevenMendezChipatec/Steven/Hiro/NLP/brain/data/Tabla_Tareas.csv")
    # Write
    df = read_file.append({'Dia': j, 'Fecha': day_list, 'Frase': frase, 'Fecha_Tarea': fecha_tarea}, ignore_index=True)
    # Save
    df.to_csv('data/Tabla_Tareas.csv', header=True, index=False)
    print(df)

def record_Speaker():
    import pyaudio
    import wave
    
    RESPEAKER_RATE = 16000
    RESPEAKER_CHANNELS = 1 # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
    RESPEAKER_WIDTH = 2
    # run getDeviceInfo.py to get index
    RESPEAKER_INDEX = 1  # refer to input device id
    CHUNK = 1024
    RECORD_SECONDS = 5
    # Nombre del audio, el cual se va a guardar en la carpeta del programa
    WAVE_OUTPUT_FILENAME = "data/output.wav"
    
    p = pyaudio.PyAudio()
    
    stream = p.open(
                rate=RESPEAKER_RATE,
                format=p.get_format_from_width(RESPEAKER_WIDTH),
                channels=RESPEAKER_CHANNELS,
                input=True,
                input_device_index=RESPEAKER_INDEX,)
    
    print("* recording")
    
    frames = []
    
    for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("* done recording")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(RESPEAKER_CHANNELS)
    wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
    wf.setframerate(RESPEAKER_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def record():
    duracion=7
    archivo="audio_speech_text.wav"
    audio=pyaudio.PyAudio()
    
    stream=audio.open(format=pyaudio.paInt16,
					channels=1, rate=16000,input=True,
					frames_per_buffer=320)
					
    print("Grabando...")
    frames=[]

    for i in range(0,int(16000/320*duracion)):
	    data=stream.read(320)
	    frames.append(data)
	
    print("Grabacion ha terminado")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile=wave.open(archivo,'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    waveFile.setframerate(16000)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def load_audio(wav_path):
    """Load wav file with the required format"""

    audio, _ = sf.read(wav_path, dtype="int16")
    audio = audio / np.iinfo(np.int16).max
    audio = np.expand_dims(audio, axis=0)
    audio = audio.astype(np.float32)
    return audio


# ==================================================================================================


def predict(interpreter, audio):
    """Feed an audio signal with shape [1, len_signal] into the network and get the predictions"""

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Enable dynamic shape inputs
    interpreter.resize_tensor_input(input_details[0]["index"], audio.shape)
    interpreter.allocate_tensors()

    # Feed audio
    interpreter.set_tensor(input_details[0]["index"], audio)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data


# ==================================================================================================


def feed_chunk(
    chunk: np.array, overlap: int, offset: int, interpreter, decoder
) -> None:
    """Feed an audio chunk with shape [1, len_chunk] into the decoding process"""

    # Get network prediction for chunk
    prediction = predict(interpreter, chunk)
    prediction = prediction[0]

    # Extract the interesting part in the middle of the prediction
    timesteps_overlap = int(len(prediction) / (chunk.shape[1] / overlap)) - 2
    prediction = prediction[timesteps_overlap:-timesteps_overlap]

    # Apply some offset for improved results
    prediction = prediction[: len(prediction) - offset]

    # Feed into decoder
    decoder.next(prediction.tolist())


# ==================================================================================================


def decode(decoder):
    """Get decoded prediction and convert to text"""
    results = decoder.decode(num_results=1)
    results = [(res.confidence, ds_alphabet.Decode(res.tokens)) for res in results]

    lm_text = results[0][1]
    return lm_text


# ==================================================================================================


def streamed_transcription(interpreter, wav_path):
    """Transcribe an audio file chunk by chunk"""

    # For reasons of simplicity, a wav-file is used instead of a microphone stream
    audio = load_audio(wav_path)
    audio = audio[0]

    # Add some empty padding that the last words are not cut from the transcription
    audio = np.concatenate([audio, np.zeros(shape=frame_overlap, dtype=np.float32)])

    start = 0
    buffer = np.zeros(shape=2 * frame_overlap + chunk_size, dtype=np.float32)
    while start < len(audio):

        # Cut a chunk from the complete audio signal
        stop = min(len(audio), start + chunk_size)
        chunk = audio[start:stop]
        start = stop

        # Add new frames to the end of the buffer
        buffer = buffer[chunk_size:]
        buffer = np.concatenate([buffer, chunk])

        # Now feed this frame into the decoding process
        ibuffer = np.expand_dims(buffer, axis=0)
        feed_chunk(ibuffer, frame_overlap, char_offset, interpreter, ds_decoder)

    # Get the text after the stream is finished
    global text
    text = decode(ds_decoder)
    print("Prediction scorer: {}".format(text))

# ==================================================================================================


def main():
    # Grabar y guardar el audio como output.wav
    record_wav = record_Speaker()

    # Cargar el modelo
    print("\nLoading model ...")
    interpreter = tflite.Interpreter(model_path=checkpoint_file, num_threads=mp.cpu_count())

    # Pasar el audio al modelo, el cual retorna un texto
    print("Running transcription ...\n")
    streamed_transcription(interpreter, test_wav_path)
    
    # Saber si la frase es consultar o agendar
    consultar_agendar(text)

# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")