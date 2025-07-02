{
  pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
  },
}:

pkgs.mkShell rec {
  name = "asr-server";
  buildInputs = with pkgs; [
    (pkgs.python3.withPackages (
      python-pkgs: with python-pkgs; [
        numpy
        transformers
        fastapi
        uvicorn
        ctranslate2
      ]
    ))

    ffmpeg-full
    python311
    stdenv.cc.cc.lib
    stdenv.cc
    cudaPackages.cudatoolkit
    linuxPackages.nvidia_x11
    zlib # Common dependency
  ];

  CUDA_PATH = pkgs.cudaPackages.cudatoolkit;
  EXTRA_LDFLAGS = "-L${pkgs.linuxPackages.nvidia_x11}/lib";
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;

  shellHook = ''
    echo "Setting up environment for asr-server with CUDA..."

    # Set the locale.
    export LC_ALL="en_US.UTF-8"
    export LANG="en_US.UTF-8"
    export PYTHONIOENCODING="utf-8"

    # Set CUDA variables
    export CUDA_VISIBLE_DEVICES=0
    export XDG_CACHE_HOME="$HOME/.cache"

    # Create and activate virtual environment
    if [ ! -d ".venv" ]; then
      echo "Creating Python virtual environment..."
      ${pkgs.python311}/bin/python3.11 -m venv .venv
    else
      echo "Re-activating existing Python virtual environment..."
    fi

    # Activate the virtual environment
    source .venv/bin/activate
    echo "Virtual environment activated."

    # pip upgrade
    pip install --upgrade pip 

    # Install torch and torchaudio
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

    pip install zhconv
    pip install faster_whisper
    pip install modelscope funasr funasr_onnx

    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')"
  '';
}
