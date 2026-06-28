# Clasificación de Emociones Caninas con InceptionV3

Trabajo Práctico Final IIA (Opción 4: Fine-Tuning).
Autores: Santiago Bussanich, Tomás Castagnino, Gaspar Giménez.

Sistema que clasifica el estado emocional de un perro (`angry`, `happy`, `relaxed`, `sad`)
a partir de imágenes, usando Transfer Learning y Fine-Tuning sobre **InceptionV3**.

- Dataset: [Dog Emotion (Kaggle)](https://www.kaggle.com/datasets/danielshanbalico/dog-emotion) — 4 clases, 1000 imágenes cada una.
- Notebook principal: `TP_Final_IIA_Bussanich,_Castagnino,_Gimenez.ipynb`.

El notebook está preparado para correr **igual en Google Colab y en local**: la celda de
credenciales detecta el entorno automáticamente, no hay que tocar el código en ningún caso.

---

## Credenciales de Kaggle: ¿de dónde salen y por qué?

El dataset se baja con `kagglehub`, que necesita autenticarse contra Kaggle. Para eso hace
falta un **API token** propio de tu cuenta. Lo generás una sola vez:

1. Entrá a Kaggle → tu avatar → **Settings**.
2. Sección **API** → botón **Create New Token**.
3. Se descarga un archivo `kaggle.json` con esta forma:

   ```json
   { "username": "tu_usuario", "key": "una_clave_larga" }
   ```

> ⚠️ Ese archivo son tus credenciales: **no lo subas al repo**. Ya está en el `.gitignore`.

A partir de ahí, **de dónde lee esas credenciales depende del entorno**, y esa es la
decisión que resuelve el notebook solo:

| Entorno | De dónde toma las credenciales | Por qué |
| --- | --- | --- |
| **Local** | `kaggle.json` en la **raíz del proyecto** (junto a `requirements.txt`) | Todo lo que necesita el proyecto queda en la misma carpeta, sin depender de tu home. El archivo está en el `.gitignore`, así que no se sube al repo. |
| **Google Colab** | Secrets del notebook (`KAGGLE_USERNAME`, `KAGGLE_KEY`) | En Colab no hay un home persistente cómodo y subir el `.json` a la sesión es inseguro. Los *Secrets* quedan cifrados en tu cuenta y no se ven en el código. |

La celda de credenciales hace `try: from google.colab import userdata` → si estamos en Colab,
usa los Secrets; si el import falla (estás en local), lee el `kaggle.json` de la raíz del
proyecto y exporta `KAGGLE_USERNAME` / `KAGGLE_KEY` automáticamente.

---

## Opción A — Correr en local

### 1. Requisito: Python 3.12

TensorFlow todavía **no** publica wheels para Python 3.13/3.14. Si usás una versión más nueva,
`pip install tensorflow` falla con *"No matching distribution found"*. Usá **Python 3.12**.

En Mac con Homebrew:

```bash
brew install python@3.12
```

En Windows (PowerShell):

```powershell
winget install Python.Python.3.12
```

(o bajalo de [python.org](https://www.python.org/downloads/)).

### 2. Crear y activar el entorno virtual

**Mac / Linux:**

```bash
# parado en la carpeta del proyecto
python3.12 -m venv .venv
source .venv/bin/activate
python --version                 # debe decir 3.12.x
```

**Windows (PowerShell):**

```powershell
# parado en la carpeta del proyecto
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version                 # debe decir 3.12.x
```

> Si PowerShell bloquea el script de activación con un error de *"execution policy"*,
> corré una sola vez:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` y volvé a activar.

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Colocar las credenciales de Kaggle

Poné el `kaggle.json` (el del paso de credenciales) en la **raíz del proyecto**, al lado de
`requirements.txt`.

**Mac / Linux** (además restringí sus permisos):

```bash
# parado en la carpeta del proyecto
mv ~/Downloads/kaggle.json ./            # ajustá el origen si está en otra carpeta
chmod 600 kaggle.json                    # solo tu usuario puede leerlo
ls -l kaggle.json                        # debe empezar con -rw-------
```

**Windows (PowerShell):**

```powershell
# parado en la carpeta del proyecto
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" .\   # ajustá el origen si está en otra carpeta
```

> En Windows no hace falta `chmod`: el archivo ya queda protegido por tu cuenta de usuario.
> Lo importante es que `kaggle.json` esté en la raíz del proyecto y que **no** se suba al
> repo (ya está en el `.gitignore`).

> El notebook lee el `kaggle.json` desde el directorio de trabajo, por eso conviene lanzar
> Jupyter / VS Code **parado en la carpeta del proyecto** (es lo normal de todos modos).

### 5. Abrir el notebook

```bash
jupyter notebook
```

Se abre el navegador, clic en el `.ipynb` y corré las celdas con `Shift + Enter`.

> Alternativa: abrir el `.ipynb` directo en **VS Code** (extensiones Python + Jupyter) y
> elegir el `.venv` como kernel.

**Notas de local:**
- La primera vez, `kagglehub` baja el dataset a `~/.cache/kagglehub/` y después lo reutiliza.
  La ruta que imprime `path` cambia respecto a Colab, pero el código no, porque siempre se usa
  la variable `path`.
- En Mac con Apple Silicon, el paquete `tensorflow` ya trae soporte (no hace falta
  `tensorflow-macos`). Entrena en CPU: para este dataset está bien, solo no abuses de las épocas.
- **En Windows entrena en CPU**: TF no soporta GPU en Windows nativo (>= 2.11). Para este
  dataset la CPU alcanza. Si tenés una GPU NVIDIA y querés acelerarlo, mirá la
  **Opción C — GPU local con WSL2**.

---

## Opción B — Correr en Google Colab

1. Subí el `.ipynb` a Colab (o abrilo desde GitHub/Drive).
2. Cargá tus credenciales como **Secrets** (icono de 🔑 en la barra izquierda). Creá dos:
   - `KAGGLE_USERNAME` → el `username` del `kaggle.json`.
   - `KAGGLE_KEY` → el `key` del `kaggle.json`.
   - Activá el toggle "Notebook access" en ambos.
3. Corré las celdas de arriba hacia abajo. La detección de entorno usa esos Secrets sola.

> En Colab no hace falta crear `.venv` ni instalar `requirements.txt`: el entorno ya viene
> con casi todo, y la primera celda hace `!pip install kagglehub`.

---

## Opción C — GPU local con WSL2 (Windows + NVIDIA)

> **Por qué WSL2:** TensorFlow **dejó de soportar GPU en Windows nativo** a partir de la
> versión 2.11 (este proyecto usa 2.21). En Windows nativo, TF **siempre** corre en CPU,
> aunque tengas CUDA instalado. La única forma de usar la GPU con TF moderno en Windows es
> a través de **WSL2** (un Ubuntu dentro de Windows). En la Opción A (Windows nativo) el
> entrenamiento funciona igual, solo que en CPU.

Requisitos: una **GPU NVIDIA** y el **driver de Windows actualizado** (el driver trae el
soporte CUDA-on-WSL; **no** se instala driver dentro de Linux).

### 1. Instalar WSL2 con Ubuntu

```powershell
wsl --install -d Ubuntu      # si ya tenés WSL, saltealo
wsl --status                 # debe decir "Versión predeterminada: 2"
```

Verificá que Ubuntu vea la GPU:

```powershell
wsl -d Ubuntu -- nvidia-smi  # debe listar tu GPU
```

### 2. Crear el entorno dentro de WSL (Python 3.12)

Ubuntu reciente trae Python 3.14 por defecto, demasiado nuevo para TF 2.21. Usamos
[`uv`](https://docs.astral.sh/uv/) para tener Python 3.12 sin tocar el sistema. **Dentro de
WSL** (`wsl -d Ubuntu` y después estos comandos en la terminal de Ubuntu):

```bash
# instalar uv si no lo tenés
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# crear el venv con Python 3.12 (en el filesystem de Linux, NO en /mnt/c)
uv venv --python 3.12 ~/venvs/tp-final

# instalar TF con CUDA + el resto de dependencias
uv pip install --python ~/venvs/tp-final/bin/python \
  "tensorflow[and-cuda]==2.21.0" keras==3.15.0 kagglehub==1.0.2 \
  pandas==3.0.3 numpy==2.5.0 pillow==12.2.0 scikit-learn==1.9.0 \
  matplotlib==3.11.0 seaborn==0.13.2 opencv-python==4.13.0.92 ipykernel
```

> El `[and-cuda]` baja las librerías CUDA/cuDNN como paquetes pip (no hace falta instalar
> CUDA aparte). El venv va en el home de Linux (`~/venvs/...`) para que sea rápido y para
> que OneDrive no intente sincronizar miles de archivos.

### 3. Registrar el kernel de Jupyter (con las libs CUDA en el PATH)

TF no encuentra las librerías CUDA salvo que `LD_LIBRARY_PATH` apunte a ellas. Para que el
kernel las tenga siempre, lo registramos con esa variable incrustada. **Dentro de WSL:**

```bash
VENV=~/venvs/tp-final
SP=$VENV/lib/python3.12/site-packages

# registrar el kernel
$VENV/bin/python -m ipykernel install --user --name tp-final-gpu \
  --display-name "Python (TP-FINAL GPU/WSL)"

# armar LD_LIBRARY_PATH con todas las libs nvidia + libcuda de WSL e inyectarlo al kernel
LDP=$(echo $SP/nvidia/*/lib | tr ' ' ':'):/usr/lib/wsl/lib
KJSON=~/.local/share/jupyter/kernels/tp-final-gpu/kernel.json
$VENV/bin/python - "$KJSON" "$LDP" <<'PY'
import json, sys
kjson, ldp = sys.argv[1], sys.argv[2]
k = json.load(open(kjson))
k.setdefault("env", {})
k["env"]["LD_LIBRARY_PATH"] = ldp
k["env"]["TF_CPP_MIN_LOG_LEVEL"] = "1"
json.dump(k, open(kjson, "w"), indent=1)
print("kernel.json listo")
PY
```

Comprobá que la GPU se usa:

```bash
~/venvs/tp-final/bin/python -c \
  'import os; os.environ["LD_LIBRARY_PATH"]=":".join(__import__("glob").glob(os.path.expanduser("~/venvs/tp-final/lib/python3.12/site-packages/nvidia/*/lib")))+":/usr/lib/wsl/lib"; import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
# debe imprimir: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 4. Usarlo desde VS Code

1. Instalá la extensión **WSL** de Microsoft.
2. `Ctrl+Shift+P` → **"WSL: Reopen Folder in WSL"** (o "Connect to WSL" y abrí la carpeta del
   proyecto, que desde WSL está en `/mnt/c/...`).
3. Abrí el `.ipynb` y en el selector de kernel elegí **"Python (TP-FINAL GPU/WSL)"**.
4. Corré las celdas. Confirmá con `tf.config.list_physical_devices("GPU")`.

**Notas de GPU/WSL:**
- **La primera corrida del entrenamiento es más lenta.** Las GPU muy nuevas (serie RTX 50,
  arquitectura Blackwell / compute capability 12.0) no están en los binarios precompilados de
  TF 2.21: TF compila los kernels desde PTX la primera vez (un warning menciona "30 minutos",
  en la práctica son segundos). Después CUDA los cachea en `~/.nv/ComputeCache` y va a full.
  **No es un error.**
- El dataset se baja a `~/.cache/kagglehub` **dentro de WSL** (rápido). El `kaggle.json` se lee
  de la carpeta del proyecto en `/mnt/c`, igual que en la Opción A.
- Te quedan dos entornos: el `.venv` de Windows (CPU) y el de WSL (GPU). Para GPU, usá siempre
  el kernel **"...GPU/WSL"**.

---

## Módulo de inferencia en tiempo real — `app.py`

Una vez entrenado el modelo desde el notebook, podés usarlo directamente con la cámara:

```bash
# con el .venv activo
python app.py
```

Al arrancar:
1. Lista los archivos `.keras` que haya en la carpeta `models/`.
2. Te pide que elijas uno por número.
3. Abre la cámara (índice 0 por defecto) y muestra la clasificación en tiempo real.
4. Presioná **`q`** o **Esc** para cerrar.

> Los modelos deben estar en `models/` con extensión `.keras`.
> El notebook los guarda ahí al terminar el entrenamiento.

---

## Estructura del proyecto

```
.
├── TP_Final_IIA_Bussanich,_Castagnino,_Gimenez.ipynb   # notebook principal
├── app.py                                              # inferencia en tiempo real (cámara)
├── models/                                             # modelos .keras entrenados
├── requirements.txt                                    # dependencias (local)
├── .gitignore
└── README.md
```

El dataset, el `.venv` y los modelos entrenados (`*.keras`) **no** se versionan: se regeneran
corriendo el notebook.
