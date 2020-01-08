# plexus-prediction-function

# dev setup VSOnline
gcloud init
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
cho 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
pyenv
pyenv install 3.7.6
pyenv global
pyenv global 3.7.6
pyenv versions

pipenv --python 3.7.3 install

# Gcloud functions deploy with storage trigger
gcloud functions deploy predict --runtime python37 --trigger-resource plexus-1d216.appspot.com --trigger-event google.storage.object.finalize --set-env-vars BLURRED_BUCKET_NAME=plexus-1d216-media-processed --memory=512MB


# docker build
gcloud builds submit --tag gcr.io/plexus-1d216/plexus-prediction-function