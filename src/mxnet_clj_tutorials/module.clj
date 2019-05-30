(ns mxnet-clj-tutorials.module
  "Tutorial for the `module` API."
  (:require
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
    [org.apache.clojure-mxnet.symbol :as sym]
    [org.apache.clojure-mxnet.visualization :as viz]))

;; Inspiration from: https://mxnet.incubator.apache.org/api/clojure/module.html
;; The Module API lets us train/optimize a Neural Network symbol

(def sample-size 1000)
(def train-size 800)
(def valid-size (- sample-size train-size))

(def feature-count 100)
(def category-count 10)
(def batch-size 10)

;;; Generating the Data Set

(def X
  (random/uniform 0 1 [sample-size feature-count]))

(def Y
  (-> sample-size
      (repeatedly #(rand-int category-count))
      (ndarray/array [sample-size])))

;; Checking X and Y data

(ndarray/shape-vec X) ;[1000 100]
(take 10 (ndarray/->vec X)) ;(0.36371076 0.32504722 0.57019675 0.038425427 0.43860152 0.63427407 0.9883738 0.95894927 0.102044806 0.6527903)

(ndarray/shape-vec Y) ;[1000]
(take 10 (ndarray/->vec Y)) ;(2.0 0.0 8.0 2.0 7.0 9.0 1.0 0.0 0.0 5.0)

;;; Splitting the Data Set in train/valid - 80/20

(def X-train
  (ndarray/crop X
                (mx-shape/->shape [0 0])
                (mx-shape/->shape [train-size feature-count])))

(def X-valid
  (ndarray/crop X
                (mx-shape/->shape [train-size 0])
                (mx-shape/->shape [sample-size feature-count])))

(def Y-train
  (ndarray/crop Y
                (mx-shape/->shape [0])
                (mx-shape/->shape [train-size])))

(def Y-valid
  (ndarray/crop Y
                (mx-shape/->shape [train-size])
                (mx-shape/->shape [sample-size])))

;; Checking train and valid data

(ndarray/shape-vec X-train) ;[800 100]
(take 10 (ndarray/->vec X-train)) ;(0.36371076 0.32504722 0.57019675 0.038425427 0.43860152 0.63427407 0.9883738 0.95894927 0.102044806 0.6527903)
(ndarray/shape-vec X-valid) ;[200 100]
(take 10 (ndarray/->vec X-valid)) ;(0.39140648 0.85629326 0.17789091 0.6476683 0.11563718 0.3868664 0.6273503 0.017593056 0.11406484 0.62723494)
(ndarray/shape-vec Y-train) ;[800]
(take 10 (ndarray/->vec Y-train)) ;(9.0 1.0 8.0 8.0 6.0 3.0 1.0 2.0 4.0 9.0)
(ndarray/shape-vec Y-valid) ;[200]
(take 10 (ndarray/->vec Y-valid)) ;(4.0 3.0 7.0 8.0 2.0 2.0 7.0 3.0 9.0 2.0)

;;; Building the Network as a symbolic graph of computations

(defn get-symbol []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden 128})
    (sym/activation "act1" {:data data :act-type "relu"})
    (sym/fully-connected "fc2" {:data data :num-hidden category-count})
    (sym/softmax-output "softmax" {:data data})))


;;; Building the Data Iterator

(def train-iter
  (mx-io/ndarray-iter [X-train]
                      {:label-name "softmax_label"
                       :label [Y-train]
                       :data-batch-size batch-size}))

(def valid-iter
  (mx-io/ndarray-iter [X-valid]
                      {:label-name "softmax_label"
                       :label [Y-valid]
                       :data-batch-size batch-size}))

;; Wrapping the computation graph in a `module`
(def model-module (m/module (get-symbol)))

;;; Training the Model

(defn train! [model-module]
  (-> model-module
      (m/bind {:data-shapes (mx-io/provide-data train-iter)
               :label-shapes (mx-io/provide-label train-iter)})
      ;; Initializing weights with Xavier
      (m/init-params {:initializer (initializer/xavier)})
      ;; Choosing Optimizer Algorithm: SGD with lr = 0.1
      (m/init-optimizer {:optimizer (optimizer/sgd {:learning-rate 0.1})})
      ;; Training for `num-epochs`
      (m/fit {:train-data train-iter :eval-data valid-iter :num-epoch 50})))

(train! model-module)
; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Train-accuracy=0.105
; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Time cost=275
; INFO  org.apache.mxnet.module.BaseModule: Epoch[0] Validation-accuracy=0.09
; INFO  org.apache.mxnet.module.BaseModule: Epoch[1] Train-accuracy=0.12125
; INFO  org.apache.mxnet.module.BaseModule: Epoch[1] Time cost=154
; INFO  org.apache.mxnet.module.BaseModule: Epoch[1] Validation-accuracy=0.095
; INFO  org.apache.mxnet.module.BaseModule: Epoch[2] Train-accuracy=0.14625
; INFO  org.apache.mxnet.module.BaseModule: Epoch[2] Time cost=134
; INFO  org.apache.mxnet.module.BaseModule: Epoch[2] Validation-accuracy=0.095
; INFO  org.apache.mxnet.module.BaseModule: Epoch[3] Train-accuracy=0.165
; INFO  org.apache.mxnet.module.BaseModule: Epoch[3] Time cost=135
; INFO  org.apache.mxnet.module.BaseModule: Epoch[3] Validation-accuracy=0.095
; INFO  org.apache.mxnet.module.BaseModule: Epoch[4] Train-accuracy=0.19875
; ...
; ...
; INFO  org.apache.mxnet.module.BaseModule: Epoch[47] Train-accuracy=1.0
; INFO  org.apache.mxnet.module.BaseModule: Epoch[47] Time cost=122
; INFO  org.apache.mxnet.module.BaseModule: Epoch[47] Validation-accuracy=0.09
; INFO  org.apache.mxnet.module.BaseModule: Epoch[48] Train-accuracy=1.0
; INFO  org.apache.mxnet.module.BaseModule: Epoch[48] Time cost=122
; INFO  org.apache.mxnet.module.BaseModule: Epoch[48] Validation-accuracy=0.085
; INFO  org.apache.mxnet.module.BaseModule: Epoch[49] Train-accuracy=1.0
; INFO  org.apache.mxnet.module.BaseModule: Epoch[49] Time cost=126
; INFO  org.apache.mxnet.module.BaseModule: Epoch[49] Validation-accuracy=0.09

;; Wow! Training accuracy is 1.0 -> It got 100% of training data right!

;;; Validating the Model

(m/score model-module
         {:eval-data valid-iter
          :eval-metric (eval-metric/accuracy)}) ;["accuracy" 0.09]

;; Really bad!
;; Of course the model cannot generalize because it is random data!
;; The data is completely meaningless.
;; The model will not be able to predict anything!

;;; Saving the Model to disk

(def save-prefix "my-model")

(m/save-checkpoint model-module
                   {:prefix save-prefix
                    :epoch 50
                    :save-opt-states true})

(def model-module-2
  (m/load-checkpoint {:prefix save-prefix
                      :epoch 50
                      :load-optimizer-states true}))

;; One can now resume training or start predicting with `model-module-2`
