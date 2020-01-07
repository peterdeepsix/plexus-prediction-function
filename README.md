# plexus-prediction-function

# dev setup VSOnline
gcloud init

# Gcloud functions deploy with storage trigger
gcloud functions deploy make_thumbnail --runtime python37 --trigger-resource plexus-1d216.appspot.com --trigger-event google.storage.object.finalize