(ns mxnet-clj-tutorials.finetune
  (:require [clojure.string :as str]

            [org.apache.clojure-mxnet.initializer :as init]
            [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.module :as m]
            [org.apache.clojure-mxnet.eval-metric :as eval-metric]
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

;; ImageRecordIter for training
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

;; ImageRecordIter for validation
(defonce val-iter
  (mx-io/image-record-iter
    {:path-imgrec valid-rec
     :data-name "data"
     :label-name "softmax_label"
     :batch-size batch-size
     :data-shape data-shape}))

(defn get-model!
  "Loads pretrained model given a `model-name`.

  Ex:
    (get-model! \"resnet-18\")"
  [model-name]
  (let [mod (m/load-checkpoint {:prefix (str model-dir "/" model-name) :epoch 0})]
    {:msymbol (m/symbol mod)
     :arg-params (m/arg-params mod)
     :aux-params (m/aux-params mod)}))

(defn mk-fine-tune-model
  "Makes the fine tune symbol `net` given the pretrained network `msymbol`.

   `msymbol`: the pretrained network symbol
   `arg-params`: the argument parameters of the pretrained model
   `num-classes`: the number of classes for the fine-tune datasets
   `layer-name`: the layer name before the last fully-connected layer"
  [{:keys [msymbol arg-params num-classes layer-name]
    :or {layer-name "flatten0"}}]
  (let [all-layers (sym/get-internals msymbol)
        net (sym/get all-layers (str layer-name "_output"))]
    {:net (as-> net data
            ;; Adding a classifier head to the base network `net`
            (sym/fully-connected "fc1" {:data data :num-hidden num-classes})
            (sym/softmax-output "softmax" {:data data}))
     :new-args (->> arg-params
                    (remove (fn [[k v]] (str/includes? k "fc1")))
                    (into {}))}))

(defn fit!
  "Trains the symbol `net` on `devs` with `train-iter` for training and
  `val-iter` for validation."
  [devs net arg-params aux-params num-epoch train-iter val-iter]
  (-> net
      ;; Converting the `net` symbol to a `module`
      (m/module {:contexts devs})
      ;; Binding data and labels for training
      (m/bind {:data-shapes (mx-io/provide-data-desc train-iter)
               :label-shapes (mx-io/provide-label-desc val-iter)})
      ;; Initializing parameters and auxiliary states
      (m/init-params {:arg-params arg-params
                      :aux-params aux-params
                      :allow-missing true})
      ;; Training the module
      (m/fit {:train-data train-iter
              :eval-data val-iter
              :num-epoch num-epoch
              :fit-params
              (m/fit-params
                {:eval-metric (eval-metric/accuracy)
                 :intializer (init/xavier {:rand-type "gaussian"
                                           :factor-type "in"
                                           :magnitude 2})
                 :batch-end-callback (callback/speedometer batch-size 10)})})))

(defn fine-tune!
  "Fine tunes `model` on `devs` for `num-epoch` with `train-iter` for training
  and `val-iter` for validation."
  ([model num-epoch devs]
   (fine-tune! model num-epoch devs train-iter val-iter))
  ([model num-epoch devs train-iter val-iter]
   (let [{:keys [msymbol arg-params aux-params]} model
         {:keys [net new-args]} (mk-fine-tune-model
                                  (assoc model :num-classes num-classes))]
     (fit! devs net new-args arg-params num-epoch train-iter val-iter))))

(comment

  (require '[mxnet-clj-tutorials.finetune :refer :all])
  (require '[org.apache.clojure-mxnet.context :as context])

  ;; On CPU: will be very slow
  (fine-tune! (get-model! "resnet-18") 1 [(context/cpu)])
  ; 64% training accuracy in about 30min of training
  ; 87% validation accuracy
  ; Can feed on average 2.5 images per second to the model for training

  ; ...
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [540]       Speed: 2.43 samples/secTrain-accuracy=0.626063
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [550]       Speed: 2.33 samples/secTrain-accuracy=0.629401
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [560]       Speed: 2.09 samples/secTrain-accuracy=0.632799
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [570]       Speed: 2.37 samples/secTrain-accuracy=0.635201
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [580]       Speed: 2.20 samples/secTrain-accuracy=0.638210
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [590]       Speed: 2.39 samples/secTrain-accuracy=0.641117
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Train-accuracy=0.64111674
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Time cost=2248485
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Validation-accuracy=0.87297297

  (fine-tune! (get-model! "resnet-18") 6 [(context/gpu)])
  ; 91% training accuracy in less than 2min of training
  ; 91% validation accuracy
  ; Can feed on average 170 images per second to the model for training

  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [60]        Speed: 169.31 samples/sec       Train-accuracy=0.920338
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [70]        Speed: 168.73 samples/sec       Train-accuracy=0.919454
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [80]        Speed: 169.76 samples/sec       Train-accuracy=0.919560
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [90]        Speed: 170.03 samples/sec       Train-accuracy=0.918441
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Train-accuracy=0.91796875
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Time cost=34549
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Validation-accuracy=0.91032606

  (fine-tune! (get-model! "resnet-50") 5 [(context/gpu)])
  ; On a bigger model: Resnet50
  ; 98.1% training accuracy in less than 5min of training
  ; 94.4% validation accuracy
  ; SOTA 2018!!
  ; Can feed on average 49 images per second to the model for training

  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [10]        Speed: 48.33 samples/sec       Train-accuracy=0.071023
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[0] Batch [20]        Speed: 48.57 samples/sec       Train-accuracy=0.132440
  ; ...
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [170]       Speed: 48.66 samples/sec       Train-accuracy=0.981542
  ; INFO  org.apache.mxnet.Callback$Speedometer: Epoch[5] Batch [180]       Speed: 47.69 samples/sec       Train-accuracy=0.981354
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Train-accuracy=0.9814189
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Time cost=119747
  ; INFO  org.apache.mxnet.module.BaseModule: Epoch[5] Validation-accuracy=0.9436141

  )
