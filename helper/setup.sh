rm -r ml-playground > /dev/null 2>&1
git clone https://github.com/daystram/ml-playground.git

pip install pyvirtualdisplay > /dev/null 2>&1
apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1

apt-get update > /dev/null 2>&1
apt-get install cmake > /dev/null 2>&1
pip install --upgrade setuptools 2>&1
pip install ez_setup > /dev/null 2>&1

pip install -r requirements.txt > /dev/null 2>&1
