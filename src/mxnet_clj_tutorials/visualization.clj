(ns mxnet-clj-tutorials.visualization
  "Functions and utils to render pretrained and user defined models."
  (:require
    [org.apache.clojure-mxnet.module :as m]
    [org.apache.clojure-mxnet.visualization :as viz]

    [mxnet-clj-tutorials.lenet :as lenet]))

;; Run the `download_vgg16.sh` and `download_resnet18.sh`
;; prior to running the following code

(def model-dir "model")
(def model-render-dir "model_render")

;; Loading pretrained models

(def vgg16-mod
  "VGG16 Module"
  (m/load-checkpoint {:prefix (str model-dir "/vgg16") :epoch 0}))

(def resnet18-mod
  "Resnet18 Module"
  (m/load-checkpoint {:prefix (str model-dir "/resnet-18") :epoch 0}))

(def resnet152-mod
  "Resnet152 Module"
  (m/load-checkpoint {:prefix (str model-dir "/resnet-152") :epoch 0}))

(def inception-mod
  "Inception Module"
  (m/load-checkpoint {:prefix (str model-dir "/Inception-BN") :epoch 0}))

(defn render-model!
  "Render the `model-sym` and saves it as a pdf file in `path/model-name.pdf`"
  [{:keys [model-name model-sym input-data-shape path]}]
  (let [dot (viz/plot-network
              model-sym
              {"data" input-data-shape}
              {:title model-name
               :node-attrs {:shape "oval" :fixedsize "false"}})]
    (viz/render dot model-name path)))

(comment
  ;; Run the following function calls to render the models in `model-render-dir`

  ;; Rendering pretrained VGG16
  (render-model! {:model-name "vgg16"
                  :model-sym (m/symbol vgg16-mod)
                  :input-data-shape [1 3 244 244]
                  :path model-render-dir})

  ;; Rendering pretrained Resnet18
  (render-model! {:model-name "resnet18"
                  :model-sym (m/symbol resnet18-mod)
                  :input-data-shape [1 3 244 244]
                  :path model-render-dir})

  ;; Rendering pretrained Resnet152
  (render-model! {:model-name "resnet152"
                  :model-sym (m/symbol resnet152-mod)
                  :input-data-shape [1 3 244 244]
                  :path model-render-dir})

  ;; Rendering pretrained Inception
  (render-model! {:model-name "inception"
                  :model-sym (m/symbol inception-mod)
                  :input-data-shape [1 3 244 244]
                  :path model-render-dir})

  ;; Rendering user defined LeNet
  (render-model! {:model-name "lenet"
                  :model-sym (lenet/get-symbol)
                  :input-data-shape [1 3 28 28]
                  :path model-render-dir}))
