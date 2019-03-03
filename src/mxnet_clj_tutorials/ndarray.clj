(ns mxnet-clj-tutorials.ndarray
  "Tutorial for `ndarray` manipulations"
  (:require
    [org.apache.clojure-mxnet.dtype :as d]
    [org.apache.clojure-mxnet.ndarray :as ndarray]
    [org.apache.clojure-mxnet.image :as mx-img]
    [org.apache.clojure-mxnet.random :as random]
    [org.apache.clojure-mxnet.shape :as mx-shape]

    [opencv4.core :as cv]
    [opencv4.utils :as cvu]))

;; Create an `ndarray` with content set to a specific value
(def a
  (ndarray/array [1 2 3 4 5 6] [2 3]))

;; Getting the dtype
(ndarray/dtype a) ;#object[scala.Enumeration$Val 0x781578fc "float32"]

;; Getting the shape as a clojure vector
(ndarray/shape-vec a) ;[2 3]

;; Visualizing the ndarray as a Clojure vector
(ndarray/->vec a) ;[1.0 2.0 3.0 4.0 5.0 6.0]

;; Ndarray creations
(let [b (ndarray/zeros [2 5])
      c (ndarray/ones [1 3 24 24])]
  (println (ndarray/dtype b)) ;#object[scala.Enumeration$Val 0x781578fc float32]
  (println (ndarray/shape-vec b)) ;[2 5]
  (println (ndarray/->vec b)) ;[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
  (println (ndarray/dtype c)) ;#object[scala.Enumeration$Val 0x781578fc float32]
  (println (ndarray/shape-vec c)) ;[1 3 24 24]
  (println (ndarray/->vec c)) ;[1.0 1.0 ... 1.0]
  )

;; Cast to another dtype
(let [b (ndarray/as-type a d/INT32)]
  (println (ndarray/dtype b)) ;#object[scala.Enumeration$Val 0x7364f96f int32]
  (println (ndarray/->vec b)) ;[1.0 2.0 3.0 4.0 5.0 6.0]
  )

;; Arithmetic Operations
(let [b (ndarray/ones [1 5])
      c (ndarray/zeros [1 5])]
  (println (ndarray/->vec (ndarray/+ b c))) ;[1.0 1.0 1.0 1.0 1.0]
  (println (ndarray/->vec (ndarray/* b c))) ;[0.0 0.0 0.0 0.0 0.0]
  )

(let [b (ndarray/array (range 1 (inc (* 3 3 3))) [3 3 3])]
  (ndarray/shape-vec (ndarray/slice b 1))
  )

;; Slice Operations
(let [b (ndarray/array (range 1 (inc (* 3 2 2))) [3 2 2]) ; Pretend b is an RGB image of size 2x2
      b0 (ndarray/slice b 0) ;; Retrieving the red pixel values
      b1 (ndarray/slice b 1)  ;; Retrieving the green pixel values
      b2 (ndarray/slice b 1 3)] ;; Retrieving the green and blue pixel values

  (println (ndarray/->vec b0)) ;[1.0 2.0 3.0 4.0]
  (println (ndarray/shape-vec b0)) ;[1 2 2]

  (println (ndarray/->vec b1)) ;[5.0 6.0 7.0 8.0]
  (println (ndarray/shape-vec b1)) ;[1 2 2]

  (println (ndarray/->vec b2)) ;[5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0]
  (println (ndarray/shape-vec b2)) ;[2 2 2]
  )

;; Transposition
(let [at (ndarray/transpose a)]
  (println (ndarray/shape-vec a)) ;[2 3]
  (println (ndarray/shape-vec at)) ;[3 2]
  )

;; Matrix operations
(let [b (ndarray/transpose a)
      c (ndarray/dot a b)
      d (ndarray/dot b a)]
  (println (ndarray/->vec c)) ;[14.0 32.0 32.0 77.0]
  (println (ndarray/shape-vec c)) ;[2 2]
  (println (ndarray/->vec d)) ;[17.0 22.0 27.0 22.0 29.0 36.0 27.0 36.0 45.0]
  (println (ndarray/shape-vec d)) ;[3 3]
  )

;; Initializing random ndarrays
(let [u (ndarray/uniform 0 1 (mx-shape/->shape [2 2]))
      g (ndarray/normal 0 1 (mx-shape/->shape [2 2]))]

  ;; Uniform Distribution
  (println (ndarray/shape-vec u)) ;[2 2]
  (println (ndarray/->vec u)) ;[0.94374806 0.9025985 0.6818203 0.44994998]

  ;; Gaussian Distribution
  (println (ndarray/shape-vec g)) ;[2 2]
  (println (ndarray/->vec g)) ;[1.2662556 0.8950642 -0.6015945 1.2040559]
  )

;; Initializing random ndarrays with `random`
(let [u (random/uniform 0 1 [2 2])
      g (random/normal 0 1 [2 2])]

  ;; Uniform Distribution
  (println (ndarray/shape-vec u)) ;[2 2]
  (println (ndarray/->vec u)) ;[0.94374806 0.9025985 0.6818203 0.44994998]

  ;; Gaussian Distribution
  (println (ndarray/shape-vec g)) ;[2 2]
  (println (ndarray/->vec g)) ;[1.2662556 0.8950642 -0.6015945 1.2040559]
  )

;;; Loading Data as NDArrays

;; With mxnet `image` API
(let [img-filename "images/mnist_digit_8.jpg"  ;; MNIST digit 8 sample
      img-grayscale-nd (mx-img/read-image img-filename {:color-flag 0})
      img-color-nd (mx-img/read-image img-filename {:color-flag 1})]
  ;; Grayscale image
  (println (ndarray/shape-vec img-grayscale-nd)) ;[28 28 1]
  (println (ndarray/dtype img-grayscale-nd)) ;#object[scala.Enumeration$Val 0x328889c6 uint8]

  ;; Color Image
  (println (ndarray/shape-vec img-color-nd)) ;[28 28 3]
  (println (ndarray/dtype img-color-nd)) ;#object[scala.Enumeration$Val 0x328889c6 uint8]
  )

;; With OpenCV
(let [img-filename "images/mnist_digit_8.jpg"  ;; MNIST digit 8 sample
      img-grayscale-nd  (-> img-filename
                            (cv/imread cv/COLOR_BGR2GRAY)
                            cvu/mat->flat-rgb-array
                            (ndarray/array [28 28 1]))
      img-color-nd (-> img-filename
                       cv/imread
                       cvu/mat->flat-rgb-array
                       (ndarray/array [28 28 3]))]
  ;; Grayscale Image
  (println (ndarray/shape-vec img-grayscale-nd)) ;[28 28 1]
  (println (ndarray/dtype img-grayscale-nd)) ;#object[scala.Enumeration$Val 0x328889c6 float]

  ;; Color Image
  (println (ndarray/shape-vec img-color-nd)) ;[28 28 3]
  (println (ndarray/dtype img-color-nd)) ;#object[scala.Enumeration$Val 0x328889c6 uint8]
  )

;; Note:
;; * OpenCV default for channel ordering is `BGR`
;; * mxnet default for channel ordering is `RGB`
