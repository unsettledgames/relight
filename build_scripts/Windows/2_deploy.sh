#!/bin/bash
# This is a script shell for deploying a relight-portable folder.
# Requires a properly built relight (see 1_build.sh).
#
# Without given arguments, the folder that will be deployed is relight/install.
#
# You can give as argument the path where relight is installed.

SCRIPTS_PATH="$(dirname "$(realpath "$0")")"
INSTALL_PATH=$SCRIPTS_PATH/../../install

#checking for parameters
for i in "$@"
do
case $i in
    -i=*|--install_path=*)
        INSTALL_PATH="${i#*=}"
        shift # past argument=value
        ;;
    *)
          # unknown option
    ;;
esac
done

windeployqt $INSTALL_PATH/relight.exe

if [ ! -f $INSTALL_PATH/vc_redist.x64.exe ]
then
    echo "Downloading vc_redist because it was missing..."
    wget https://aka.ms/vs/17/release/vc_redist.x64.exe --output-document=$INSTALL_PATH/vc_redist.x64.exe
fi

#at this point, distrib folder contains all the files necessary to execute meshlab
echo "$INSTALL_PATH is now a self contained relight application"