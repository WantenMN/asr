{
  pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
  },
}:

pkgs.mkShell rec {
  name = "asr-client";
  buildInputs = with pkgs; [
    (pkgs.python3.withPackages (
      python-pkgs: with python-pkgs; [
        keyboard
        requests
        pyperclip
        numpy
        pyaudio
        sounddevice
        webrtcvad
        scipy
        setuptools
        pydub
        speechrecognition
      ]
    ))
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
}
