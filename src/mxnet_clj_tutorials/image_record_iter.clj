(ns mxnet-clj-tutorials.image-record-iter
  "Tutorial for ImageRecordIter API."
  (:require [org.apache.clojure-mxnet.io :as mx-io]
            [org.apache.clojure-mxnet.ndarray :as ndarray]
            [opencv4.mxnet :as mx-cv]
            [opencv4.core :as cv]
            [opencv4.utils :as cvu]))

;; Parameters
(def batch-size 10)
(def data-shape [3 224 224])
(def train-rec "data/data_train.rec")

(def train-iter
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

(defn visualize-image-rec-iter!
  ([image-rec-iter]
   (visualize-image-rec-iter! image-rec-iter 5))
  ([image-rec-iter k]
   (let [nda-data (first (mx-io/iter-data train-iter))
         mats (map (fn [i]
                     (-> nda-data
                         ;; ith image in batch
                         (ndarray/slice i)
                         (ndarray/reshape data-shape)
                         ;; Swapping [c w h] -> [w h c]
                         (ndarray/swap-axis 0 2)
                         (ndarray/swap-axis 0 1)
                         (mx-cv/ndarray-to-mat)
                         ;; Conversion BGR -> RGB
                         (cv/cvt-color! cv/COLOR_BGR2RGB)))
                   (range k))]
     (doseq [mat mats]
       (cvu/imshow mat)))
   (mx-io/reset image-rec-iter)))

(comment

  (visualize-image-rec-iter! train-iter 10))
