{
  pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
  },
}:

pkgs.mkShell rec {
  name = "asr-server";
  buildInputs = with pkgs; [
    uv
    stdenv.cc.cc.lib
    cudaPackages.cudatoolkit
    linuxPackages.nvidia_x11
  ];

  CUDA_PATH = pkgs.cudaPackages.cudatoolkit;
  EXTRA_LDFLAGS = "-L${pkgs.linuxPackages.nvidia_x11}/lib";
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;

  shellHook = ''
    uv sync

    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(uv run python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'):"
  '';
}
