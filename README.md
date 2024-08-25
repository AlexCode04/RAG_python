Ejecutar el script con python everioment:

1. crear el entorno con este comando:

python -m venv chatbot_rag_env

2. Activar el entorno:

chatbot_rag_env\Scripts\activate

3. Una vez activado el entorno instalamos las dependencias:

pip install -r requirements.txt

4. Ejecutamos el script:

python chatbot_rag.py

----------------------------EJECUTAR EL SCRIPT USANDO CONDA---------------------------

1. Instalar conda en nuestros equipos (recordar instalarlo en las variables de entorno PATH)

https://docs.anaconda.com/miniconda/

2. Una vez instalado ejecutamos el siguiente comando (este comando instalara automaticamente las dependencias):

conda env create -f environment.yml

3. Una vez ejecutado activamos el entorno:

conda activate chatbot_rag

4. Ejecutamos el script:

python chatbot_rag.py



