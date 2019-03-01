(ns mxnet-clj-tutorials.pretrained
  "Tutorial on pretrained models in mxnet: Inception, Resnet and VGG."
  (:require
    [clojure.string :as string]

    [opencv4.core :as cv]
    [opencv4.utils :as cvu]
    [org.apache.clojure-mxnet.dtype :as d]
    [org.apache.clojure-mxnet.eval-metric :as eval-metric]
    [org.apache.clojure-mxnet.executor :as executor]
    [org.apache.clojure-mxnet.initializer :as initializer]
    [org.apache.clojure-mxnet.io :as mx-io]
    [org.apache.clojure-mxnet.module :as m]
    [org.apache.clojure-mxnet.ndarray :as ndarray]
    [org.apache.clojure-mxnet.optimizer :as optimizer]
    [org.apache.clojure-mxnet.random :as random]
    [org.apache.clojure-mxnet.shape :as mx-shape]
    [org.apache.clojure-mxnet.symbol :as sym]))

;;; Loading the Model

(def model-dir "model/")

(def h 224)
(def w 224)
(def c 3)

(def inception-mod
  "Pretrained Inception BN model loaded from disk"
  (-> {:prefix (str model-dir "Inception-BN") :epoch 0}
      m/load-checkpoint
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
      (m/bind {:for-training false
               :data-shapes [{:name "data" :shape [1 c h w]}]})))

(def image-net-labels
  "ImageNet 1000 Labels"
  (-> (slurp (str model-dir "/synset.txt"))
      (string/split #"\n")))

(assert (= 1000 (count image-net-labels)))

;; Preparing the Data

(defn preprocess-img-mat
  "Preprocessing steps on an `img-mat` from OpenCV to feed into the Model"
  [img-mat]
  (-> img-mat

      ;; Resize image to (w, h)
      (cv/resize! (cv/new-size w h))

      ;; Scaling the bytes to match the new format:
      ;; [-128, 128] instead of [0, 255]
      (cv/convert-to! cv/CV_8SC3 0.5)

      ;; Substract mean pixel values from ImageNet dataset
      (cv/add! (cv/new-scalar -103.939 -116.779 -123.68))

      ;; Flatten matrix
      cvu/mat->flat-rgb-array

      ;; Reshape to (1, c h, w)
      (ndarray/array [1 c h w])))

;;; Predicting

(defn predict
  "Predict with `model` the top `k` labels from `labels` of the ndarray `nda`"
  ([model labels nda]
   (predict model labels nda 5))
  ([model labels nda k]
   (let [prob
         ;; The computation graph of the model is ran forward once
         (-> model (m/forward {:data [nda]}) m/outputs ffirst ndarray/->vec)

         prob-with-labels
         (mapv (fn [p l] {:prob p :label l}) prob labels)]
     (->> prob-with-labels
          (sort-by :prob)
          reverse
          (take k)))))

(predict inception-mod
         image-net-labels
         (preprocess-img-mat (cv/imread "images/guitarplayer.jpg")))
;({:prob 0.68194896, :label "n04296562 stage"}
;{:prob 0.06861413, :label "n03272010 electric guitar"}
;{:prob 0.04886661, :label "n10565667 scuba diver"}
;{:prob 0.044686787, :label "n03250847 drumstick"}
;{:prob 0.029348794, :label "n02676566 acoustic guitar"})

(predict inception-mod
         image-net-labels
         (preprocess-img-mat (cv/imread "images/cat.jpg")))
;({:prob 0.5226559, :label "n02119789 kit fox, Vulpes macrotis"}
;{:prob 0.14540964, :label "n02112018 Pomeranian"}
;{:prob 0.13845555, :label "n02119022 red fox, Vulpes vulpes"}
;{:prob 0.06784552, :label "n02120505 grey fox, gray fox, Urocyon cinereoargenteus"}
;{:prob 0.024868377, :label "n02441942 weasel"})

(predict inception-mod
         image-net-labels
         (preprocess-img-mat (cv/imread "images/dog.jpg")))
;({:prob 0.89285797, :label "n02110958 pug, pug-dog"}
;{:prob 0.06376573, :label "n04409515 tennis ball"}
;{:prob 0.01919549, :label "n03942813 ping-pong ball"}
;{:prob 0.014978847, :label "n02108422 bull mastiff"}
;{:prob 0.0012790044, :label "n02808304 bath towel"})
