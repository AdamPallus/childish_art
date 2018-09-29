

#Set up data and split into training/validation/test ----
original_dataset_dir <- "C:/Users/User/Documents/deeplearning/abstract/abstract_original"
# 
base_dir <- "C:/Users/User/Documents/deeplearning/abstract/abstract_small"
dir.create(base_dir)
# 
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)
# 
train_child_dir <- file.path(train_dir, "child")
dir.create(train_child_dir)
# 
train_artist_dir <- file.path(train_dir, "artist")
dir.create(train_artist_dir)
# 
validation_child_dir <- file.path(validation_dir, "child")
dir.create(validation_child_dir)
# 
validation_artists_dir <- file.path(validation_dir, "artist")
dir.create(validation_artists_dir)
# 
test_child_dir <- file.path(test_dir, "child")
dir.create(test_child_dir)
# 
test_artists_dir <- file.path(test_dir, "artist")
dir.create(test_artists_dir)


# 
# from<-list.files(paste0(original_dataset_dir,'/child/'),full.names=TRUE)
# to.name<- paste0(original_dataset_dir,'child/child',1:length(from),'.jpg')
# file.rename(from=from,to.name)

# setwd(paste0(original_dataset_dir,'/artist'))
# 
# from<-list.files()
# to.name<- paste0('artist',1:length(from),'.jpg')
# file.rename(from=from,to.name)
# 
# 
# setwd(paste0(original_dataset_dir,'/child'))
# 
# from<-list.files()
# to.name<- paste0('child',1:length(from),'.jpg')
# file.rename(from=from,to.name)
# 
# setwd('~/deeplearning')

childnames<-list.files(paste0(original_dataset_dir,'/child'))
artnames<- list.files(paste0(original_dataset_dir,'/artist'))

childnames<-sample(childnames)
artnames<-sample(artnames)

fnames <- childnames[1:1000]

fail<-file.copy(file.path(paste0(original_dataset_dir,'/child'), fnames),
          file.path(train_child_dir))

fnames <- childnames[1001:1300]
fail<-file.copy(file.path(paste0(original_dataset_dir,'/child'), fnames),
          file.path(validation_child_dir))

fnames <- childnames[1301:length(childnames)]
fail<-file.copy(file.path(paste0(original_dataset_dir,'/child'), fnames),
          file.path(test_child_dir))

fnames <- artnames[1:1000]
fail<-file.copy(file.path(paste0(original_dataset_dir,'/artist'), fnames),
          file.path(train_artist_dir))

fnames <- artnames[1001:1300]
fail<-file.copy(file.path(paste0(original_dataset_dir,'/artist'), fnames),
          file.path(validation_artists_dir))

fnames <- artnames[1301:length(artnames)]
fail<-file.copy(file.path(paste0(original_dataset_dir,'/artist'), fnames),
          file.path(test_artists_dir))
# 
# fnames <- paste0("artist", 631:670, ".jpg")
# file.copy(file.path(paste0(original_dataset_dir,'/artist'), fnames),
#           file.path(train_artist_dir))

#set up model----


library(keras)

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

model <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")


freeze_weights(conv_base)

# create the base pre-trained model
base_model <- application_inception_v3(weights = 'imagenet', include_top = FALSE)

# add custom layers
predictions <- base_model$output %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(units = 1024, activation = 'relu') %>%
  layer_dropout(0.50) %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)


freeze_weights(base_model)

# 
train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
# 
# train_datagen = image_data_generator(
#   rescale = 1/255,
#   rotation_range = 90,
#   width_shift_range = 0.4,
#   height_shift_range = 0.4,
#   shear_range = 0.4,
#   zoom_range = 0.4,
#   horizontal_flip = TRUE,
#   fill_mode = "nearest"
# )

# Note that the validation data shouldn't be augmented!
test_datagen <- image_data_generator(rescale = 1/255)  

train_generator <- flow_images_from_directory(
  train_dir,                  # Target directory  
  train_datagen,              # Data generator
  target_size = c(150, 150),  # Resizes all images to 150 x 150
  batch_size = 20,
  class_mode = "binary"       # binary_crossentropy loss for binary labels
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 20,
  validation_data = validation_generator,
  validation_steps = 50,
  verbose=1
)

plot(history)

############
##SECOND TRAINING
##########

freeze_weights(base_model, from = 1, to = 172)
unfreeze_weights(base_model, from = 173)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = c("accuracy")
)


# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model %>% compile(
  optimizer = optimizer_sgd(lr = 0.00001, momentum = 0.9), 
  loss = 'binary_crossentropy',
  metrics = c("accuracy")
)

history2 <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 50,
  validation_data = validation_generator,
  validation_steps = 50
)

save_model_hdf5(model,'childishart-vgg3.h5')

plot(history2)

test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)



model %>% evaluate_generator(test_generator, steps = 50)


plot_prediction<-function(img_path){
  require(magick)
  img <- image_load(img_path, target_size = c(150,150))
  x <- image_to_array(img)
  # ensure we have a 4d tensor with single element in the batch dimension,
  # the preprocess the input for prediction using resnet50
  x <- array_reshape(x, c(1, dim(x)))
  x <- imagenet_preprocess_input(x)
  
  # x<- inception_v3_preprocess_input(x)
  
  # make predictions then decode and print them
  preds <- model %>% predict(x)
  
  # cat(paste0('prediction:',preds[1],'\n\n'))
  
  mystery<- image_read(img_path)
  
  if(preds[1]>0.5){
    pred_text=paste0('CHILDISH: ',round(preds[1]*100),'%')
    font='Comic Sans'
    # cat('THIS IS A artist')
    # plot(as.raster(image_to_array(img)/255))
  } else{
    pred_text=paste0('ART! ',round(100-preds[1]*100),'%')
    font='Helvetica'
    # cat('It must be a cat')
    # plot(as.raster(image_to_array(img)/255))
  }
  mystery %>%
    image_scale(500) %>%
    image_annotate(pred_text,size=50,
                   boxcolor='gray80',
                   color='gray20',
                   gravity='southwest',
                   font=font)
}


save_predictions<-function(img_path,infolder,outfolder){
  pimage<-plot_prediction(paste0(infolder,img_path))
  image_write(pimage, path = paste(outfolder,img_path,sep=""))
}


#collect----
get_prediction<-function(img_path){
  
  img = tryCatch({
    img <- image_load(img_path, target_size = c(150,150))
  }, warning = function(w) {
    return(NULL)
  }, error = function(e) {
    img<-NULL
  })
  
  if(is.null(img)) return(-1)
  if(class(img)[1]!='PIL.Image.Image') return(-1)
  x <- image_to_array(img)
  # ensure we have a 4d tensor with single element in the batch dimension,
  # the preprocess the input for prediction using resnet50
  x <- array_reshape(x, c(1, dim(x)))
  # x <- imagenet_preprocess_input(x)
  
  x<- inception_v3_preprocess_input(x)
  
  # make predictions then decode and print them
  preds <- model %>% predict(x)
  return(preds[1])
  
}


path="abstract/abstract_small/test/artist"

path=file.path('F:/web files/Google Images/2018-1-29 18-35-41')
path=file.path('F:\\web files\\Google Images\\2018-1-29 18-36-23')

dirs<-list.dirs('F:\\web files\\Google Images')
dirs<- dirs[2:21]

# for (i in seq_along(dirs)){
for (i in 10:length(dirs)){
  path<-dirs[i]
  cat(paste0('Analyzing...',path,'\n'))
  files <- list.files(path=path,pattern='*.jpg')
  
  fp<- sapply(files,function(x) get_prediction(file.path(path,x)))
  
  group1<- files[fp>0.8]
  group2<-files[fp<=0.2&fp>0]
  group3<- files[fp>0.2&fp<=0.8]
  
  cat(paste0('Child: ',length(group1),
            ' || Art: ',length(group2),
            ' || Unsure: ',length(group3),'\n'))
  
  result.child<-file.copy(file.path(path, group1),
                          file.path('autochild'))
  
  result.artist<-file.copy(file.path(path, group2),
                           file.path('autoartist'))
  
  result.artist<-file.copy(file.path(path, group3),
                           file.path('autounsure'))
}



