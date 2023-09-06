import requests
import os
from bs4 import BeautifulSoup

root = '/home/bcostanza/MachineLearning/project/data/'

# URL de la página web
url_pagina = "https://users.flatironinstitute.org/~fvillaescusa/VOSS5/"

# Realiza la solicitud HTTP a la página
response = requests.get(url_pagina)

# Verifica si la solicitud fue exitosa
if response.status_code == 200:
    # Obtiene el directorio de destino donde se guardarán las carpetas y los archivos
    directorio_destino = root

    soup = BeautifulSoup(response.content, "html.parser")

    enlaces = soup.find_all("a")

    # Recorre los enlaces y descarga los archivos
    
    for enlace in enlaces:
        ruta_enlace = enlace["href"]
        if ruta_enlace.endswith("/"):
            if ruta_enlace != '../':
                
                nombre_folder = ruta_enlace.rstrip("/").split("/")[-1]
                ruta_folder = os.path.join(directorio_destino, nombre_folder)

                if not os.path.exists(ruta_folder):
                    os.makedirs(ruta_folder)

            
                response_folder = requests.get(url_pagina + nombre_folder)
                if response_folder.status_code == 200:
                # Analiza el contenido HTML de la carpeta
                    soup_carpeta = BeautifulSoup(response_folder.content, "html.parser")

                    # Encuentra todos los enlaces a archivos en la carpeta
                    enlaces_archivos = soup_carpeta.find_all("a", href=True)
                    for enlace_archivo in enlaces_archivos:
                        
                        ruta_archivo = enlace_archivo["href"]
                        if not ruta_archivo.endswith("/"):
                            nombre_archivo = ruta_archivo.split("/")[-1]
                            ruta_archivo_local = os.path.join(ruta_folder, nombre_archivo)
                            print(ruta_archivo_local)
                            # Descarga y guarda el archivo
                            response_archivo = requests.get(url_pagina + ruta_enlace + ruta_archivo)
                            if response_archivo.status_code == 200:
                                with open(ruta_archivo_local, 'wb') as archivo:
                                    archivo.write(response_archivo.content)


              

   

print("Descarga completa.")

   

    
