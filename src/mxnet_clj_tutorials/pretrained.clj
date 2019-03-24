(ns mxnet-clj-tutorials.pretrained
  "Tutorial on pretrained models with MXNet: Inception, ResNet and VGG."
  (:require
    [clojure.string :as string]

    [opencv4.core :as cv]
    [opencv4.utils :as cvu]

    [org.apache.clojure-mxnet.module :as m]
    [org.apache.clojure-mxnet.ndarray :as ndarray]))

;;; Loading the Models

(def model-dir "model/")

(def h 224) ;; Image height
(def w 224) ;; Image width
(def c 3)   ;; Number of channels: Red, Green, Blue

;; Pretrained Inception BN model loaded from disk
(defonce inception-mod
  (-> {:prefix (str model-dir "Inception-BN") :epoch 0}
      (m/load-checkpoint)
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
      (m/bind {:for-training false
               :data-shapes [{:name "data" :shape [1 c h w]}]})))

(defonce resnet-152-mod
  (-> {:prefix (str model-dir "resnet-152") :epoch 0}
      (m/load-checkpoint)
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
      (m/bind {:for-training false
               :data-shapes [{:name "data" :shape [1 c h w]}]})))

(defonce vgg-16-mod
  (-> {:prefix (str model-dir "vgg16") :epoch 0}
      (m/load-checkpoint)
      ;; Define the shape of input data and bind the name of the input layer
      ;; to "data"
      (m/bind {:for-training false
               :data-shapes [{:name "data" :shape [1 c h w]}]})))

;; ImageNet 1000 Labels

(defonce image-net-labels
  (-> (str model-dir "/synset.txt")
      (slurp)
      (string/split #"\n")))

(assert (= 1000 (count image-net-labels)))

;;; Preparing the Data

(defn preprocess-img-mat
  "Preprocessing steps on an `img-mat` from OpenCV to feed into the Model"
  [img-mat]
  (-> img-mat
      ;; Resize image to (w, h)
      (cv/resize! (cv/new-size w h))
      ;; Maps pixel values from [-128, 128] to [0, 127]
      (cv/convert-to! cv/CV_8SC3 0.5)
      ;; Substract mean pixel values from ImageNet dataset
      (cv/add! (cv/new-scalar -103.939 -116.779 -123.68))
      ;; Flatten matrix
      (cvu/mat->flat-rgb-array)
      ;; Reshape to (1, c, h, w)
      (ndarray/array [1 c h w])))

;;; Predicting

(defn- top-k
  "Return top `k` from prob-maps with :prob key"
  [k prob-maps]
  (->> prob-maps
       (sort-by :prob)
       (reverse)
       (take k)))

(defn predict
  "Predict with `model` the top `k` labels from `labels` of the ndarray `x`"
  ([model labels x]
   (predict model labels x 5))
  ([model labels x k]
   (let [probs (-> model
                   (m/forward {:data [x]})
                   (m/outputs)
                   (ffirst)
                   (ndarray/->vec))
         prob-maps (mapv (fn [p l] {:prob p :label l}) probs labels)]
     (top-k k prob-maps))))

(comment
  (->> "images/guitarplayer.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))
  ;({:prob 0.68194896, :label "n04296562 stage"}
  ;{:prob 0.06861413, :label "n03272010 electric guitar"}
  ;{:prob 0.04886661, :label "n10565667 scuba diver"}
  ;{:prob 0.044686787, :label "n03250847 drumstick"}
  ;{:prob 0.029348794, :label "n02676566 acoustic guitar"})

  (->> "images/guitarplayer.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))
  ;({:prob 0.31067622, :label "n03272010 electric guitar"}
  ;{:prob 0.14873363, :label "n04296562 stage"}
  ;{:prob 0.04211086, :label "n04141076 sax, saxophone"}
  ;{:prob 0.032480247, :label "n04536866 violin, fiddle"}
  ;{:prob 0.022555437, :label "n03110669 cornet, horn, trumpet, trump"})

  (->> "images/guitarplayer.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict resnet-152-mod image-net-labels))
  ;({:prob 0.32477784, :label "n02708093 analog clock"}
  ;{:prob 0.16388302, :label "n03388043 fountain"}
  ;{:prob 0.16345626, :label "n04286575 spotlight, spot"}
  ;{:prob 0.13510057, :label "n03028079 church, church building"}
  ;{:prob 0.046880785, :label "n03000134 chainlink fence"})

  (->> "images/guitarplayer2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))
  ;({:prob 0.647201, :label "n03272010 electric guitar"}
  ;{:prob 0.3371953, :label "n04296562 stage"}
  ;{:prob 0.008809802, :label "n02676566 acoustic guitar"}
  ;{:prob 0.0024602208, :label "n02787622 banjo"}
  ;{:prob 0.0018765739, :label "n03759954 microphone, mike"})

  (->> "images/guitarplayer2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))
  ;({:prob 0.73966444, :label "n03272010 electric guitar"}
  ;{:prob 0.105860166, :label "n04296562 stage"}
  ;{:prob 0.059584185, :label "n04141076 sax, saxophone"}
  ;{:prob 0.029627431, :label "n02787622 banjo"}
  ;{:prob 0.016049441, :label "n02676566 acoustic guitar"})

  (->> "images/guitarplayer2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict resnet-152-mod image-net-labels))
  ;({:prob 0.7606491, :label "n04286575 spotlight, spot"}
  ;{:prob 0.051547647, :label "n09229709 bubble"}52  ;{:prob 0.020731576, :label "n02708093 analog clock"}
  ;{:prob 0.017223129, :label "n03388043 fountain"}
  ;{:prob 0.017017603, :label "n04153751 screw"})

  (->> "images/cat.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))
  ;({:prob 0.5226559, :label "n02119789 kit fox, Vulpes macrotis"}
  ;{:prob 0.14540964, :label "n02112018 Pomeranian"}
  ;{:prob 0.13845555, :label "n02119022 red fox, Vulpes vulpes"}
  ;{:prob 0.06784552, :label "n02120505 grey fox, gray fox, Urocyon cinereoargenteus"}
  ;{:prob 0.024868377, :label "n02441942 weasel"})

  (->> "images/cat.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))
  ;({:prob 0.41937035, :label "n02119789 kit fox, Vulpes macrotis"}
  ;{:prob 0.26819462, :label "n02119022 red fox, Vulpes vulpes"}
  ;{:prob 0.07655225, :label "n02124075 Egyptian cat"}
  ;{:prob 0.049807232, :label "n02123159 tiger cat"}
  ;{:prob 0.034435965, :label "n02123045 tabby, tabby cat"})

  (->> "images/cat.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict resnet-152-mod image-net-labels))
  ;({:prob 0.19741254, :label "n02441942 weasel"}
  ;{:prob 0.1101544, :label "n04589890 window screen"}
  ;{:prob 0.04704221, :label "n02443484 black-footed ferret, ferret, Mustela nigripes"}
  ;{:prob 0.031224, :label "n02442845 mink"}
  ;{:prob 0.030577147, :label "n01806567 quail"})

  (->> "images/cat2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))
  ;({:prob 0.9669817, :label "n02124075 Egyptian cat"}
  ;{:prob 0.020066999, :label "n02123045 tabby, tabby cat"}
  ;{:prob 0.0071042357, :label "n02123159 tiger cat"}
  ;{:prob 0.005353994, :label "n02127052 lynx, catamount"}
  ;{:prob 4.658187E-5, :label "n02123597 Siamese cat, Siamese"})

  (->> "images/cat2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))
  ;({:prob 0.9030159, :label "n02124075 Egyptian cat"}
  ;{:prob 0.05147686, :label "n02123045 tabby, tabby cat"}
  ;{:prob 0.024212556, :label "n02123159 tiger cat"}
  ;{:prob 0.0099070445, :label "n02127052 lynx, catamount"}
  ;{:prob 3.7205187E-4, :label "n04040759 radiator"})

  (->> "images/cat2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict resnet-152-mod image-net-labels))
  ;({:prob 0.09747321, :label "n02391049 zebra"}
  ;{:prob 0.04086459, :label "n03532672 hook, claw"}
  ;{:prob 0.04000138, :label "n01773157 black and gold garden spider, Argiope aurantia"}
  ;{:prob 0.03932228, :label "n03187595 dial telephone, dial phone"}
  ;{:prob 0.038932547, :label "n02124075 Egyptian cat"})

  (->> "images/dog.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))
  ;({:prob 0.89285797, :label "n02110958 pug, pug-dog"}
  ;{:prob 0.06376573, :label "n04409515 tennis ball"}
  ;{:prob 0.01919549, :label "n03942813 ping-pong ball"}
  ;{:prob 0.014978847, :label "n02108422 bull mastiff"}
  ;{:prob 0.0012790044, :label "n02808304 bath towel"})

  (->> "images/dog.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))
  ;({:prob 0.96750915, :label "n02110958 pug, pug-dog"}
  ;{:prob 0.01833086, :label "n02108422 bull mastiff"}
  ;{:prob 0.005593519, :label "n04409515 tennis ball"}
  ;{:prob 0.0017559915, :label "n02108089 boxer"}
  ;{:prob 8.5579534E-4, :label "n02096585 Boston bull, Boston terrier"})

  (->> "images/dog.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict resnet-152-mod image-net-labels))
  ;({:prob 0.5550145, :label "n04286575 spotlight, spot"}
  ;{:prob 0.121797085, :label "n03637318 lampshade, lamp shade"}
  ;{:prob 0.10030677, :label "n03388043 fountain"}
  ;{:prob 0.06361176, :label "n02708093 analog clock"}
  ;{:prob 0.030823186, :label "n03729826 matchstick"})

  (->> "images/dog2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict inception-mod image-net-labels))
  ;({:prob 0.7363852, :label "n02110958 pug, pug-dog"}
  ;{:prob 0.23988461, :label "n02108422 bull mastiff"}
  ;{:prob 0.013495497, :label "n02108915 French bulldog"}
  ;{:prob 0.0019004685, :label "n02093428 American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier"}
  ;{:prob 0.0013417465, :label "n04409515 tennis ball"})

  (->> "images/dog2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict vgg-16-mod image-net-labels))
  ;({:prob 0.95628285, :label "n02110958 pug, pug-dog"}
  ;{:prob 0.02271582, :label "n02108422 bull mastiff"}
  ;{:prob 0.0075261267, :label "n02108915 French bulldog"}
  ;{:prob 0.0014686864, :label "n02086079 Pekinese, Pekingese, Peke"}
  ;{:prob 0.0012910544, :label "n02108089 boxer"})

  (->> "images/dog2.jpg"
       (cv/imread)
       (preprocess-img-mat)
       (predict resnet-152-mod image-net-labels))
  ;({:prob 0.26530242, :label "n01819313 sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita"}
  ;{:prob 0.10890331, :label "n03388043 fountain"}
  ;{:prob 0.04696827, :label "n03590841 jack-o'-lantern"}
  ;{:prob 0.039195884, :label "n03935335 piggy bank, penny bank"}
  ;{:prob 0.03070829, :label "n02051845 pelican"})
  )
