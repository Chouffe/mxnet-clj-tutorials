(ns mxnet-clj-tutorials.lenet
  (:require [org.apache.clojure-mxnet.symbol :as sym]))

(defn get-symbol
  "Return LeNet Symbol

  Input data shape [`batch-size` `channels` 28 28]
  Output data shape [`batch-size 10]"
  []
  (as-> (sym/variable "data") data

    ;; First `convolution` layer
    (sym/convolution "conv1" {:data data :kernel [5 5] :num-filter 20})
    (sym/activation "tanh1" {:data data :act-type "tanh"})
    (sym/pooling "pool1" {:data data :pool-type "max" :kernel [2 2] :stride [2 2]})

    ;; Second `convolution` layer
    (sym/convolution "conv2" {:data data :kernel [5 5] :num-filter 50})
    (sym/activation "tanh2" {:data data :act-type "tanh"})
    (sym/pooling "pool2" {:data data :pool-type "max" :kernel [2 2] :stride [2 2]})

    ;; Flattening before the Fully Connected Layers
    (sym/flatten "flatten" {:data data})

    ;; First `fully-connected` layer
    (sym/fully-connected "fc1" {:data data :num-hidden 500})
    (sym/activation "tanh3" {:data data :act-type "tanh"})

    ;; Second `fully-connected` layer
    (sym/fully-connected "fc2" {:data data :num-hidden 10})

    ;; Softmax Loss
    (sym/softmax-output "softmax" {:data data})))
