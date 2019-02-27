(ns mxnet-clj-tutorials.image-manipulation
  (:require [clojure.java.io :as io]
            [opencv4.core :as cv]
            [opencv4.utils :as cvu]))

(defn download!
  "Download `uri` and store it in `filename` on disk"
  [uri filename]
  (with-open [in (io/input-stream uri)
              out (io/output-stream filename)]
    (io/copy in out)))

(defn preview!
  "Preview image from `filename` and display it on the screen in a new window

  >>> (preview! \"images/cat.jpg\")
  >>> (preview! \"images/cat.jpg\" :h 300 :w 200)
  "
  ([filename]
   (preview! filename {:h 400 :w 400}))
  ([filename {:keys [h w]}]
   (-> filename
       cv/imread
       (cv/resize! (cv/new-size h w))
       cvu/imshow)))

(defn preprocess-mat
  "Preprocessing steps on a `mat` from OpenCV.
  Example of commons preprocessing tasks"
  [mat]
  (-> mat
      ;; Substract mean
      (cv/add! (cv/new-scalar 103.939 116.779 123.68))
      ;; Resize
      (cv/resize! (cv/new-size 400 400))
      ;; Maps pixel values from [-125, 125] to [0, 250]
      (cv/convert-to! cv/CV_8SC3 0.5)
      ;; TODO: add cropping?
      ))

(defn mat->ndarray
  "Converts a `mat` from OpenCV to an MXNET `ndarray`"
  [mat]
  (let [h (.height mat)
        w (.width mat)
        c (.channels mat)]
    (-> mat
        cvu/mat->flat-rgb-array
        (ndarray/array [c h w]))))

;; TODO
(defn ndarray->mat
  "Converts a `ndarray` to an OpenCV `mat`"
  [ndarray]
  ;; TODO
  )

(defn filename->ndarray!
  "Convert an image stored on disk `filename` into an `ndarray`

  `filename`: string representing the image on disk
  `shape-vec`: is the actual shape of the returned `ndarray`
  "
  [filename shape-vec]
  (-> filename
      cv/imread
      mat->ndarray))

(comment

  ;; Download a cat image from a `uri` and save it into `images/cat.jpg`
  (download! "https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/python/predict_image/cat.jpg" "images/cat.jpg")

  ;; Preview an image from disk
  (preview! "images/cat.jpg")

  ;; Preview with different size
  (preview! "images/cat.jpg" {:h 300 :w 200})

  ;; Visualize preprocessing steps
  (-> "images/cat.jpg"
      cv/imread
      preprocess-mat
      cvu/imshow))
