(require-python-interop)

(def triple+1 ^{:macro true}
  (fn* args
    (let* [x (nth args 0)
           z `(~x ~x)]
      `(+ 1 ~@z ~x))))

(def bar
  (fn* args
    (let* [x (nth args 0)
           y (nth args 1)
           a (triple+1 x)
           b (+ y 2)]
      (+ a b))))

(assert (= 8 (bar 1 2)))

(assert (= 42 (eval {:locals {'bar 42}} 'bar)))

(def pl (python/import "pathlib"))
(def Path (python/getattr pl "Path"))

(def p (Path "/tmp"))

(assert ((python/getattr p "exists")))

(assert (= [1 2 3 4] [1 2 3 4]))
(assert (= {1 2 3 4} {1 2 3 4}))
(assert (= (list 1 2 3 4) (list 1 2 3 4)))

(defn qux [[x y]]
  [y x])

(assert (= [2 1] (qux [1 2 3])))

(assert (= "123" (str nil 1 nil 2 nil 3 nil)))

(assert true "This should fail")

(defn forty-two []
  42)

(assert (= 42 (forty-two)))

(defn f1 []
  (assert false))

(defn f2 []
  (if true
    (f1)
    nil))

(defn f3 []
  (f2))

(f3)
