import json
from llama_cpp import Llama
from textwrap import indent
import time



# Cargar el mododelo
print("Cargado el modelo...")

## Vicuna 7B
#vicuna = Llama(model_path="/home/mcim/Descargas/Vicuna/ggml-vic7b-uncensored-q5_1.bin")
#Vicuna 13B
#vicuna = Llama(model_path="/home/mcim/Descargas/Vicuna/ggml-vic13b-uncensored-q5_1.bin")
#Open Assistant 30B
#vicuna = Llama(model_path="/home/mcim/Descargas/Vicuna/OpenAssistant-SFT-7-Llama-30B.ggml.q4_0.bin")
#Wizard LM
vicuna = Llama(model_path="/home/mcim/Descargas/Vicuna/koala-7B.ggmlv3.q4_0.bin")



print("Modelo cargado...")


def VicunaInference(userQuestion):
    promt = "Question: " + userQuestion + " Answer:"
    output = vicuna(promt, max_tokens = 128, temperature = 0.9, stop = ["Question:", "Q:"], echo = True)
    #print(json.dumps(output, indent = 2))
    resp = output["choices"][0]["text"]
    #print(resp)
    resp = resp.split('Answer: ')[1]
    return resp

while(True):
    inicio = time.time() # tiempo al iniciar la funcion 
    userQuestion = input("Escribe una pregunta: ")
    answer = VicunaInference(userQuestion)
    print(answer)
    fin = time.time()
    print("Tiempo total: ", fin-inicio) #Mostrar el tiempo total de procesamiento
    