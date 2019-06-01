(ns mxnet-clj-tutorials.infer
  (:require [clojure.string :as str]

            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.context :as context]
            [org.apache.clojure-mxnet.dtype :as dtype]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
            [org.apache.clojure-mxnet.image :as mx-img]
            [org.apache.clojure-mxnet.infer :as infer]
            [org.apache.clojure-mxnet.initializer :as init]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.layout :as layout]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]

            [opencv4.mxnet :as mx-cv]
            [opencv4.core :as cv]
            [opencv4.utils :as cvu]))

;;; Pre-trained Image Classifier

;; Folder path with prefix to the model: (params, symbol and synset)
(def classifier-model-path-prefix "model/resnet-18")

;; Shape of the input images to feed to the model
(def classifier-descriptors
  [{:name "data"
    :shape [1 3 224 224]
    :layout layout/NCHW  ;; (batch size, channel, height, width)
    :dtype dtype/FLOAT32}])

;; Boilerplate to create the image classifier
(def classifier-factory
  (infer/model-factory classifier-model-path-prefix
                       classifier-descriptors))

;; Image classifier
(def classifier
  (infer/create-image-classifier
    classifier-factory
    {:contexts [(context/default-context)]}))


;;; Finetuned Image Classifier

(def fine-tuned-classifier-model-path-prefix
  "model/oxford-pet/oxford-pet-finetuned-resnet-50")

;; Shape of the input images to feed to the model
(def fine-tuned-classifier-descriptors
  [{:name "data"
    :shape [1 3 224 224]
    :layout layout/NCHW  ;; (batch size, channel, height, width)
    :dtype dtype/FLOAT32}])

;; Boilerplate to create the image classifier
(def fine-tuned-classifier-factory
  (infer/model-factory fine-tuned-classifier-model-path-prefix
                       fine-tuned-classifier-descriptors))

;; Image classifier
(def fine-tuned-classifier
  (infer/create-image-classifier
    fine-tuned-classifier-factory
    {:contexts [(context/default-context)]}))


;;; Pre-trained Image detector

(def detector-model-path-prefix
  "model/resnet50_ssd/resnet50_ssd_model")

(def detector-descriptors
  [{:name "data"
    :shape [1 3 512 512]
    :layout layout/NCHW
    :dtype dtype/FLOAT32}])

(def detector-factory
  (infer/model-factory detector-model-path-prefix
                       detector-descriptors))

(def detector
  (infer/create-object-detector
    detector-factory
    {:contexts [(context/default-context)]}))

(comment

  ;;; Classification

  ;; With custom fine tuned model

  (def image-pug
    (infer/load-image-from-file "data/oxford-pet/pug/pug_1.jpg"))
  (def image-shiba-inu
    (infer/load-image-from-file "data/oxford-pet/shiba_inu/shiba_inu_1.jpg"))

  (cvu/show (cvu/buffered-image-to-mat image-pug))
  (cvu/show (cvu/buffered-image-to-mat image-shiba-inu))

  (infer/classify-image fine-tuned-classifier image-pug 5)
  ; [[{:class "pug", :prob 0.967651}
  ; {:class "american bulldog", :prob 0.0070187477}
  ; {:class "boxer", :prob 0.004099819}
  ; {:class "japanese chin", :prob 0.0018000666}
  ; {:class "stadffordshire bull terrier", :prob 0.001757793}]]

  (infer/classify-image fine-tuned-classifier image-shiba-inu 5)
  ; [[{:class "shiba inu", :prob 0.37148225}
  ; {:class "miniature pinscher", :prob 0.27084318}
  ; {:class "american pit bull terrier", :prob 0.12005531}
  ; {:class "stadffordshire bull terrier", :prob 0.057867803}
  ; {:class "chihuahua", :prob 0.022154164}]]

  ;; With pre-trained model

  (def image-dog
    (infer/load-image-from-file "data/oxford-pet/samoyed/samoyed_1.jpg"))
  (def image-cat
    (infer/load-image-from-file "data/oxford-pet/Persian/Persian_10.jpg"))

  (cvu/show (cvu/buffered-image-to-mat image-dog))
  (cvu/show (cvu/buffered-image-to-mat image-cat))

  (infer/classify-image classifier image-dog 5)
  ; [[{:class "n02111889 Samoyed, Samoyede", :prob 0.97866315}
  ; {:class "n02111500 Great Pyrenees", :prob 0.011256912}
  ; {:class "n02114548 white wolf, Arctic wolf, Canis lupus tundrarum", :prob 0.005785384}
  ; {:class "n02104029 kuvasz", :prob 8.3861506E-4}
  ; {:class "n02120079 Arctic fox, white fox, Alopex lagopus", :prob 4.5782723E-4}]]

  (infer/classify-image classifier image-cat 5)
  ; [[{:class "n02123394 Persian cat", :prob 0.9996811}
  ; {:class "n02328150 Angora, Angora rabbit", :prob 1.2610655E-4}
  ; {:class "n02127052 lynx, catamount", :prob 7.97991E-5}
  ; {:class "n02123597 Siamese cat, Siamese", :prob 1.4681199E-5}
  ; {:class "n03657121 lens cap, lens cover", :prob 1.32941805E-5}]]

  ;; Batch prediction

  (def image-dogs
    (infer/load-image-paths
      ["data/oxford-pet/samoyed/samoyed_1.jpg"
       "data/oxford-pet/samoyed/samoyed_101.jpg"
       "data/oxford-pet/beagle/beagle_1.jpg"]))

  (infer/classify-image-batch classifier image-dogs 2)
  ; [[{:class "n02111889 Samoyed, Samoyede", :prob 0.97866315}
  ; {:class "n02111500 Great Pyrenees", :prob 0.01125689}]
  ;
  ; [{:class "n02111889 Samoyed, Samoyede", :prob 0.8132068}
  ; {:class "n02104029 kuvasz", :prob 0.12649947}]]
  ;
  ; [{:class "n02088364 beagle", :prob 0.59173775}
  ; {:class "n02088238 basset, basset hound", :prob 0.34796223}]

  ;; NDArray Classification

  (infer/classify-with-ndarray
    classifier
    [(-> "data/oxford-pet/samoyed/samoyed_1.jpg"
         (mx-img/read-image)
         (mx-img/resize-image 224 224)
         ;; Swapping [w h c] -> [c w h]
         (ndarray/swap-axis 0 2)
         (ndarray/swap-axis 1 2)
         ;; Reshaping to match NCWH Layout
         (ndarray/reshape [1 3 224 224])
         ;; Converting to the right type
         (ndarray/as-type dtype/FLOAT32))]
    5)
  ; [[{:class "n02111889 Samoyed, Samoyede", :prob 0.9827139}
  ; {:class "n02111500 Great Pyrenees", :prob 0.00700274}
  ; {:class "n02114548 white wolf, Arctic wolf, Canis lupus tundrarum", :prob 0.005095501}
  ; {:class "n02120079 Arctic fox, white fox, Alopex lagopus", :prob 9.119243E-4}
  ; {:class "n02106030 collie", :prob 7.1724877E-4}]]

  (infer/classify-with-ndarray
    classifier
    [(-> "data/oxford-pet/samoyed/samoyed_1.jpg"
         (infer/load-image-from-file)
         (infer/reshape-image 224 224)
         (infer/buffered-image-to-pixels [3 224 224])
         ;; Similar to (reshape [1 3 224 224])
         (ndarray/expand-dims 0))]
    5)
  ; [[{:class "n02111889 Samoyed, Samoyede", :prob 0.9827139}
  ; {:class "n02111500 Great Pyrenees", :prob 0.00700274}
  ; {:class "n02114548 white wolf, Arctic wolf, Canis lupus tundrarum", :prob 0.005095501}
  ; {:class "n02120079 Arctic fox, white fox, Alopex lagopus", :prob 9.119243E-4}
  ; {:class "n02106030 collie", :prob 7.1724877E-4}]]

  ;;; Detector

  (def image-dog
    (infer/load-image-from-file "data/resnet50_ssd/dog.jpg"))

  (cvu/show (cvu/buffered-image-to-mat image-dog))

  (def bounding-boxes
    (first (infer/detect-objects detector image-dog 5)))
 ; [{:class "car", :prob 0.99847263, :x-min 0.60979164, :y-min 0.14068183, :x-max 0.89065313, :y-max 0.29426125}
 ; {:class "bicycle", :prob 0.904738, :x-min 0.30460563, :y-min 0.2928976, :x-max 0.7496816, :y-max 0.8182522}
 ; {:class "dog", :prob 0.8226828, :x-min 0.16371784, :y-min 0.34988278, :x-max 0.40358952, :y-max 0.9312255}
 ; {:class "bicycle", :prob 0.21815668, :x-min 0.1817387, :y-min 0.26932585, :x-max 0.46060142, :y-max 0.8074726}
 ; {:class "person", :prob 0.12772352, :x-min 0.17368972, :y-min 0.2365945, :x-max 0.31111816, :y-max 0.37116468}]

 ;; TODO: process result and draw bounding boxes

  )
