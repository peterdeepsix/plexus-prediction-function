# plexus-prediction-function

# dev setup VSOnline
gcloud init

# Gcloud functions deploy with storage trigger
gcloud functions deploy predict --runtime python37 --trigger-resource plexus-1d216.appspot.com --trigger-event google.storage.object.finalize --set-env-vars BLURRED_BUCKET_NAME=plexus-1d216-media-processed

