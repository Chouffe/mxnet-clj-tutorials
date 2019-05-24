(ns mxnet-clj-tutorials.finetune
  (:require [clojure.string :as str]

            [org.apache.clojure-mxnet.initializer :as init]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [org.apache.clojure-mxnet.symbol :as sym]
            [org.apache.clojure-mxnet.callback :as callback]
            [org.apache.clojure-mxnet.context :as context]

            [opencv4.mxnet :as mx-cv]
            [opencv4.core :as cv]
            [opencv4.utils :as cvu]))

;; Parameters
(def batch-size 10)
(def data-shape [3 224 224])
(def train-rec "data/data_train.rec")
(def valid-rec "data/data_val.rec")
(def model-dir "model")
(def num-classes 37)

(defonce train-iter
  (mx-io/image-record-iter
    {:path-imgrec train-rec
     :data-name "data"
     :label-name "softmax_label"
     :batch-size batch-size
     :data-shape data-shape

     ;; Data Augmentation
     :shuffle true  ;; Whether to shuffle data randomly or not
     ; :max-rotate-angle 50  ;; Rotate by a random degree in [-50 50]
     ; :resize 300  ;; resize the shorter edge before cropping
     :rand-crop true  ;; randomely crop the image
     :rand-mirror true  ;; randomely mirror the image
     }))

(defonce val-iter
  (mx-io/image-record-iter
    {:path-imgrec valid-rec
     :data-name "data"
     :label-name "softmax_label"
     :batch-size batch-size
     :data-shape data-shape}))

(defn get-model []
  (let [mod (m/load-checkpoint {:prefix (str model-dir "/resnet-18") :epoch 0})]
    {:msymbol (m/symbol mod)
     :arg-params (m/arg-params mod)
     :aux-params (m/aux-params mod)}))

(def model (get-model))

(def all-layers
  (sym/get-internals (:msymbol model)))

; (sym/list-outputs all-layers)

; (sym/get (sym/get-internals (:msymbol model)) "flatten0_output")


(defn get-fine-tune-model
  "msymbol: the pretrained network symbol
   arg-params: the argument parameters of the pretrained model
   num-classes: the number of classes for the fine-tune datasets
   layer-name: the layer name before the last fully-connected layer"
  [{:keys [msymbol arg-params num-classes layer-name]
    :or {layer-name "flatten0"}}]
  (let [all-layers (sym/get-internals msymbol)
        net (sym/get all-layers (str layer-name "_output"))]
    {:net (as-> net data
            (sym/fully-connected "fc1" {:data data :num-hidden num-classes})
            (sym/softmax-output "softmax" {:data data}))
     :new-args (->> arg-params
                    (remove (fn [[k v]] (str/includes? k "fc1")))
                    (into {}))}))

(defn fit [devs msymbol arg-params aux-params num-epoch]
  (let [mod (-> (m/module msymbol {:contexts devs})
                (m/bind {:data-shapes (mx-io/provide-data-desc train-iter)
                         :label-shapes (mx-io/provide-label-desc val-iter)})
                (m/init-params {:arg-params arg-params :aux-params aux-params
                                :allow-missing true}))]
    (m/fit mod
           {:train-data train-iter
            :eval-data val-iter
            :num-epoch 1
            :fit-params
            (m/fit-params
              {:intializer (init/xavier {:rand-type "gaussian"
                                         :factor-type "in"
                                         :magnitude 2})
               :batch-end-callback (callback/speedometer batch-size 10)})})))

(defn fine-tune! [devs num-epoch]
  (let [{:keys [msymbol arg-params aux-params] :as model} (get-model)
        {:keys [net new-args]} (get-fine-tune-model
                                 (merge model {:num-classes num-classes}))]
    (fit devs net new-args arg-params num-epoch)))

(comment
  ;; On CPU: will be very slow
  (fine-tune! [(context/cpu)])
  ;; On GPU: fast!!
  ; (fine-tune! [(context/gppu)])
  )
