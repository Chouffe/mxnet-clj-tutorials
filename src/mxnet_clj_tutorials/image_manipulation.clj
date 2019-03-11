(ns mxnet-clj-tutorials.image-manipulation
  "Image manipulation tutorial."
  (:require
    [clojure.java.io :as io]

    [org.apache.clojure-mxnet.image :as mx-img]
    [org.apache.clojure-mxnet.ndarray :as ndarray]
    [org.apache.clojure-mxnet.shape :as mx-shape]

    [opencv4.colors.rgb :as rgb]
    [opencv4.mxnet :as mx-cv]
    [opencv4.core :as cv]
    [opencv4.utils :as cvu])
  (:import org.opencv.core.Mat java.awt.image.DataBufferByte))

(defn download!
  "Download `uri` and store it in `filename` on disk"
  [uri filename]
  (with-open [in (io/input-stream uri)
              out (io/output-stream filename)]
    (io/copy in out)))

(defn preview!
  "Preview image from `filename` and display it on the screen in a new window
   Ex:
    (preview! \"images/cat.jpg\")
    (preview! \"images/cat.jpg\" :h 300 :w 200)"
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
      ;; Subtract mean
      (cv/add! (cv/new-scalar 103.939 116.779 123.68))
      ;; Resize
      (cv/resize! (cv/new-size 400 400))
      ;; Maps pixel values from [-125, 125] to [0, 250]
      (cv/convert-to! cv/CV_8SC3 0.5)))

(defn mat->ndarray
  "Convert a `mat` from OpenCV to an MXNet `ndarray`"
  [mat]
  (let [h (.height mat)
        w (.width mat)
        c (.channels mat)]
    (-> mat
        cvu/mat->flat-rgb-array
        (ndarray/array [c h w]))))

(defn ndarray->mat
  "Convert a `ndarray` to an OpenCV `mat`"
  [ndarray]
  (let [shape (mx-shape/->vec ndarray)
        [h w _ _] (mx-shape/->vec (ndarray/shape ndarray))
        bytes (byte-array shape)
        mat (cv/new-mat h w cv/CV_8UC3)]
    (.put mat 0 0 bytes)
    mat))

(defn filename->ndarray!
  "Convert an image stored on disk `filename` into an `ndarray`

  `filename`: string representing the image on disk
  `shape-vec`: is the actual shape of the returned `ndarray`
   return: ndarray"
  [filename shape-vec]
  (-> filename
      cv/imread
      mat->ndarray))

(defn draw-bounding-box!
  "Draw bounding box on `img` given the `top-left` and `bottom-right` coordonates.
  Add `label` when provided.
  returns: nil"
  [img {:keys [label top-left bottom-right]}]
  (let [[x0 y0] top-left
        [x1 y1] bottom-right
        top-left-point (cv/new-point x0 y0)
        bottom-right-point (cv/new-point x1 y1)]
    (cv/rectangle img top-left-point bottom-right-point rgb/white 1)
    (when label
      (cv/put-text! img label top-left-point cv/FONT_HERSHEY_DUPLEX 1.0 rgb/white 1))))

(defn draw-predictions!
    "Draw all predictions on an `img` passing `results` which is a collection
     of bounding boxes data.
     returns: nil"
    [img results]
    (doseq [{:keys [label top-left bottom-right] :as result} results]
      (draw-bounding-box! img result)))

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

(comment
  ;; Writing image to disk
  (-> "images/dog.jpg"
      ;; Convert filename to NDArray
      (mx-img/read-image {:to-rgb true}) ;; option :to-rgb, true by default
      ;; Resizing image to height = 400, width = 400
      (mx-img/resize-image 400 400)
      ;; Convert to BufferedImage
      mx-img/to-image
      ;; Saving BufferedImage to disk
      (javax.imageio.ImageIO/write "jpg" (java.io.File. "test2.jpg")))

  ;; Writing image to disk
  (-> "images/mnist_digit_8.jpg"
      ;; Load image from disk
      (mx-img/read-image {:to-rgb true})
      ;; Convert NDArray to Mat
      ndarray->mat
      ;; Save Image to disk
      (cv/imwrite "test-digit.jpg"))
      ; cvu/imshow

  ;; Showing an image using `buffered-image-to-mat`
  (-> "images/dog.jpg"
      ;; Read image from disk
      (mx-img/read-image {:to-rgb true})
      ;; Convert to BufferedImage - Can be very slow...
      mx-img/to-image
      ;; Convert to Mat
      cvu/buffered-image-to-mat
      ;; Show Mat
      cvu/imshow)

  ;; Showing an image using `ndarray->mat`
  (-> "images/dog.jpg"
      ;; Read image from disk
      (mx-img/read-image {:to-rgb false})
      ;; Convert NDArray to Mat
      ndarray->mat
      ;; Show Mat
      cvu/imshow)

  ;; Showing an image using `mx-cv/ndarray-to-mat` from `origami`
  (-> "images/dog.jpg"
      ;; Read image from disk
      (mx-img/read-image {:to-rgb false})
      ;; Convert NDArray to Mat
      mx-cv/ndarray-to-mat
      ;; Show Mat
      cvu/imshow)

  ;; Drawing one bounding box on an image of dog
  (let [img (cv/imread "images/dog.jpg")]
    (draw-bounding-box! img {:top-left [200 440]
                             :bottom-right [350 525]
                             :label "cookie"})
    (cvu/imshow img))

  ;; Drawing multiple bounding boxes on an image of dog
  (let [img (cv/imread "images/dog.jpg")
        results [{:top-left [200 70] :bottom-right [830 430] :label "dog"}
                 {:top-left [200 440] :bottom-right [350 525] :label "cookie"}]]
    (draw-predictions! img results)
    (cvu/imshow img))
  )
