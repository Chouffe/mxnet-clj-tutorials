(ns mxnet-clj-tutorials.symbol
  "Tutorial for the `symbol` API."
  (:require
    [org.apache.clojure-mxnet.context :as context]
    [org.apache.clojure-mxnet.dtype :as d]
    [org.apache.clojure-mxnet.executor :as executor]
    [org.apache.clojure-mxnet.module :as m]
    [org.apache.clojure-mxnet.ndarray :as ndarray]
    [org.apache.clojure-mxnet.symbol :as sym]
    [org.apache.clojure-mxnet.visualization :as viz]))

;;; Composing Symbols

;; Define Input data as Variable
(def a (sym/variable "A"))
(def b (sym/variable "B"))
(def c (sym/variable "C"))
(def d (sym/variable "D"))

;; Define a Computation Graph: e = (a * b) + (c * d)
(def e
  (sym/+
    (sym/* a b)
    (sym/* c d)))

;; What are the dependencies for `e`?
(sym/list-arguments e) ;["A" "B" "C" "D"]

;; What does `e` compute?
(sym/list-outputs e) ;["_plus0_output"]

;; What is the implementation of `e` as a stack of operations?
(sym/list-outputs (sym/get-internals e)) ;["A" "B" "_mul0_output" "C" "D" "_mul1_output" "_plus0_output"]

;; Render Computation Graph
(defn render-computation-graph!
  "Render the `sym` and saves it as a pdf file in `path/sym-name.pdf`"
  [{:keys [sym-name sym input-data-shape path]}]
  (let [dot (viz/plot-network
              sym
              input-data-shape
              {:title sym-name
               :node-attrs {:shape "oval" :fixedsize "false"}})]
    (viz/render dot sym-name path)))

(comment
  ;; Render the computation graph `e`
  (render-computation-graph!
    {:sym-name "e"
     :sym e
     :input-data-shape {"A" [1] "B" [1] "C" [1] "D" [1]}
     :path "model_render"}))

;;; Executing Symbols

;; Binding `ndarrays` to `symbols`
(def data-binding
  {"A" (ndarray/array [1] [1] {:dtype d/INT32})
   "B" (ndarray/array [2] [1] {:dtype d/INT32})
   "C" (ndarray/array [3] [1] {:dtype d/INT32})
   "D" (ndarray/array [4] [1] {:dtype d/INT32})})

;; Execute the graph operations `e`
(-> e
    (sym/bind data-binding)
    executor/forward
    executor/outputs
    first
    ndarray/->vec) ; We got our answer: 1 * 2 + 4 * 3 = 14

;; Execute the graph on a different device (cpu or gpu)
(-> e
    (sym/bind (context/cpu 0) data-binding)
    ; (sym/bind (context/gpu 0) data-binding)
    executor/forward
    executor/outputs
    first
    ndarray/->vec) ; We got our answer: 1 * 2 + 4 * 3 = 14

;;; Serialization - json format

(let [symbol-filename "symbol-e.json"]
  ;; Saving to disk symbol `e`
  (sym/save e symbol-filename)
  ;; Loading from disk symbol `e`
  (let [e2 (sym/load symbol-filename)]
    (println (= (sym/to-json e) (sym/to-json e2))) ;true
    ))
