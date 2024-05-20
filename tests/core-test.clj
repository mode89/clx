(in-ns 'core-test)

(import* inspect)
(import* types)
(import* pytest)

(defmacro deftest [tname & body]
  `(defn ~(symbol (str "test-" tname)) []
     ~@body))

(defmacro is [x]
  `(assert ~x "Test failed"))

(defmacro raises [ex-type arg0 & args]
  (if (string? arg0)
    `(let [pattern# ~arg0]
       (python/with [_ (python* pytest/raises
                         "(" ~ex-type ", match=" pattern# ")")]
         ~@args))
    `(python/with [_ (pytest/raises ~ex-type)]
       ~arg0
       ~@args)))

(defn range*
  ([end] (range* 0 end))
  ([start end]
   (lazy-seq
     (when (< start end)
       (cons start (range* (+ start 1) end))))))

(def FORTY-TWO 42)

(deftest fn
  (is (= 42 (eval '((fn [] 42)))))
  (is (= 9002 (eval '(do (def f (fn inc [x] (+ x 1)))
                         (f 9001)))))
  (is (= 9 (eval '(do (def foo
                        (fn [x y]
                          (+ x 3)
                          (+ y 7)))
                      (foo 1 2)))))
  (let [foo (eval '(fn ([] 42)
                       ([x] (+ x 1))
                       ([x y] (+ x y))
                       ([x y & zs] (apply list x y zs))))]
    (is (= 42 (foo)))
    (is (= 2 (foo 1)))
    (is (= 7 (foo 3 4)))
    (is (= '(5 6 7 8) (foo 5 6 7 8)))))

(deftest defn
  (is (= 3 (eval '(do (defn add [x y] (+ x y))
                      (add 1 2)))))
  (is (= 25 (eval '(do (defn bar [x y]
                         (+ x 10)
                         (+ y 20))
                       (bar 3 5)))))
  (let [foo (eval '(defn foo
                     ([] 42)
                     ([x] (+ x 1))
                     ([x y] (+ x y))
                     ([x y & zs] (apply list x y zs))))]
    (is (= 42 (foo)))
    (is (= 2 (foo 1)))
    (is (= 7 (foo 3 4)))
    (is (= '(5 6 7 8) (foo 5 6 7 8)))))

(deftest defmacro
  (eval '(defmacro if* [pred then else]
           (list 'cond pred then true else)))
  (is (= 42 (eval '(if* true 42 (throw (Exception))))))
  (eval '(defmacro plus
           ([] 0)
           ([x] x)
           ([x y] `(let [x# ~x y# ~y] (python* x# " + " y#)))
           ([x y & more] `(plus (plus ~x ~y) ~@more))))
  (is (= [0 1 5 15 45] (eval '[(plus)
                               (plus 1)
                               (plus 2 3)
                               (plus 4 5 6)
                               (plus 7 8 9 10 11)]))))

(deftest throw
  (raises Exception "Hello, World!"
    (throw (Exception "Hello, World!"))))

(deftest let
  (is (= 42 (let [x 42] x)))
  (is (= 9011 (let [x 42
                    y 9001]
                (+ x y)
                (+ y 10)))))

(deftest set!
  (is (= 42 (let [foo (types/SimpleNamespace)]
              (set! foo bar 42)
              (.-bar foo)))))

(deftest if
  (is (= 42 (if true 42 (throw (Exception)))))
  (is (= 43 (if false (throw (Exception)) 43))))

(deftest if-not
  (is (= 42 (if-not false 42 (throw (Exception)))))
  (is (= 43 (if-not true (throw (Exception)) 43))))

(deftest when
  (is (= 42 (when true 42)))
  (is (= nil (when false (throw (Exception))))))

(deftest when-not
  (is (= 42 (when-not false 42)))
  (is (= nil (when-not true (throw (Exception))))))

(deftest when-let
  (is (= 42 (when-let [x 42] x)))
  (is (= nil (when-let [x nil] (throw (Exception))))))

(deftest assert
  (is (= 42 (do (assert true) 42)))
  (raises Exception
    (assert false))
  (raises Exception "Hello.*World!"
    (assert false "Hello, World!")))

(deftest lazy-seq
  (is (= '() (lazy-seq nil)))
  (is (= '(1 2 3) (lazy-seq '(1 2 3))))
  (is (= 4 (letfn [(foo [x]
                     (lazy-seq (cons x (foo (+ x 1)))))]
             (first (next (next (next (foo 1))))))))
  (let [s (lazy-seq
            (cons 42
              (lazy-seq
                (cons 9001
                  (lazy-seq
                    (throw (Exception)))))))]
    (is (= 42 (first s)))
    (is (= 9001 (first (rest s))))
    (raises Exception
      (first (rest (rest s))))
    (is (= 9001 (first (next s))))
    (raises Exception
      (next (next s)))))

(deftest nil?
  (is (= true (nil? nil)))
  (is (= false (nil? false)))
  (is (= false (nil? true)))
  (is (= false (nil? 42)))
  (is (= false (nil? :hello)))
  (is (= false (nil? '())))
  (is (= false (nil? [])))
  (is (= false (nil? (python/list)))))

(deftest string?
  (is (= false (string? nil)))
  (is (= false (string? false)))
  (is (= false (string? true)))
  (is (= false (string? 42)))
  (is (= false (string? :hello)))
  (is (= false (string? '())))
  (is (= false (string? [])))
  (is (= false (string? (python/list))))
  (is (= true (string? "hello"))))

(deftest some?
  (is (= false (some? nil)))
  (is (= true (some? false)))
  (is (= true (some? true)))
  (is (= true (some? 42)))
  (is (= true (some? :hello)))
  (is (= true (some? '())))
  (is (= true (some? [])))
  (is (= true (some? (python/list)))))

(deftest not
  (is (= true (not nil)))
  (is (= true (not false)))
  (is (= false (not true)))
  (is (= false (not 42)))
  (is (= false (not :hello))))
  ; TODO (is (= false (not '())))
  ; TODO (is (= false (not []))
  ; TODO (is (= false (not (python/list)))))

(deftest even?
  (is (= true (even? 0)))
  (is (= false (even? 1)))
  (is (= true (even? 2)))
  (is (= false (even? 3))))

(deftest odd?
  (is (= false (odd? 0)))
  (is (= true (odd? 1)))
  (is (= false (odd? 2)))
  (is (= true (odd? 3))))

(deftest inc
  (is (= 1 (inc 0)))
  (is (= 2 (inc 1)))
  (is (= 3 (inc 2)))
  (is (= -1 (inc -2))))

(deftest dec
  (is (= -1 (dec 0)))
  (is (= 0 (dec 1)))
  (is (= 1 (dec 2)))
  (is (= -3 (dec -2))))

(deftest zero?
  (is (= true (zero? 0)))
  (is (= false (zero? 1)))
  (is (= false (zero? 2)))
  (is (= false (zero? -1))))

(deftest pos?
  (is (= false (pos? 0)))
  (is (= true (pos? 1)))
  (is (= true (pos? 2)))
  (is (= false (pos? -1))))

(deftest neg?
  (is (= false (neg? 0)))
  (is (= false (neg? 1)))
  (is (= false (neg? 2)))
  (is (= true (neg? -1))))

(deftest operators
  (is (= 3 (+ 1 2)))
  (is (= -1 (- 3 4)))
  (is (= 30 (* 5 6)))
  (is (= 0.875 (/ 7 8)))
  (is (= false (= 9 10)))
  (is (= true (= 11 11)))
  (is (= true (< 12 13)))
  (is (= false (< 14 14)))
  (is (= false (< 16 15)))
  (is (= true (<= 17 18)))
  (is (= true (<= 19 19)))
  (is (= false (<= 21 20)))
  (is (= false (> 22 23)))
  (is (= false (> 24 24)))
  (is (= true (> 26 25)))
  (is (= false (>= 27 28)))
  (is (= true (>= 29 29)))
  (is (= true (>= 31 30))))

(deftest and
  (is (= true (and)))
  (is (= true (and true)))
  (is (= false (and false)))
  (is (= true (and true true)))
  (is (= false (and true false)))
  (is (= false (and false true)))
  (is (= false (and false false)))
  (is (= true (and true true true)))
  (is (= false (and true true false)))
  (is (= false (and true false true)))
  (is (= false (and true false false)))
  (is (= false (and false true true)))
  (is (= false (and false true false)))
  (is (= false (and false false true)))
  (is (= false (and false false false)))
  (is (= nil (and nil)))
  (is (= nil (and true nil)))
  (is (= nil (and nil false)))
  (is (= false (and false nil)))
  (is (= nil (and nil nil)))
  (is (= nil (and true true nil))))

(deftest or
  (is (= nil (or)))
  (is (= true (or true)))
  (is (= false (or false)))
  (is (= true (or true true)))
  (is (= true (or true false)))
  (is (= true (or false true)))
  (is (= false (or false false)))
  (is (= true (or true true true)))
  (is (= true (or true true false)))
  (is (= true (or true false true)))
  (is (= true (or true false false)))
  (is (= true (or false true true)))
  (is (= true (or false true false)))
  (is (= true (or false false true)))
  (is (= false (or false false false)))
  (is (= nil (or nil)))
  (is (= true (or true nil)))
  (is (= false (or nil false)))
  (is (= nil (or false nil)))
  (is (= nil (or nil nil)))
  (is (= nil (or false false nil)))
  (is (= false (or false nil false))))


(deftest map
  (is (= '() (map inc nil)))
  (is (= '(2 3 4) (map inc '(1 2 3))))
  (is (= '(2 3 4) (map inc [1 2 3])))
  (let [s (map (fn [x]
                 (if (< x 3)
                   x
                   (throw (Exception))))
               [1 2 3])]
    (is (= 1 (first s)))
    (is (= 2 (first (next s))))
    (raises Exception
      (next (next s))))
  (let [ls (map (fn [x] (* x 3))
                (range* 10000000))]
    (is (= 0 (first ls)))
    (is (= 3 (first (next ls))))))

(deftest filter
  (is (= '() (filter nil nil)))
  (is (= '() (filter odd? nil)))
  (is (= '(1 3) (filter odd? '(1 2 3))))
  (is (= '(1 3) (filter odd? [1 2 3])))
  (let [s (filter (fn [x]
                    (if (< x 3)
                      (odd? x)
                      (throw (Exception))))
                  [1 2 3])]
    (is (= 1 (first s)))
    (raises Exception
      (next s)))
  (let [ls (filter odd? (range* 10000000))]
    (is (= 1 (first ls)))
    (is (= 3 (first (next ls))))
    (is (= 5 (first (next (next ls)))))))

(deftest reduce
  (is (= nil (reduce nil nil nil)))
  (is (= 42 (reduce + '(42))))
  (is (= 6 (reduce + '(1 2 3))))
  (is (= 15 (reduce + [4 5 6])))
  (is (= 34 (reduce + 7 '(8 9 10))))
  (is (= (+ 42 (python/sum (python/range 10)))
         (reduce (fn [x y]
                   (+ x y))
                 42
                 (range* 10)))))

(deftest take
  (is (= '() (take 0 nil)))
  (is (= '() (take 0 '(1 2 3))))
  (is (= '() (take 0 [1 2 3])))
  (is (= '(1) (take 1 '(1 2 3))))
  (is (= '(1) (take 1 [1 2 3])))
  (is (= '(1 2) (take 2 '(1 2 3))))
  (is (= '(1 2) (take 2 [1 2 3])))
  (is (= '(1 2 3) (take 3 '(1 2 3))))
  (is (= '(1 2 3) (take 3 [1 2 3])))
  (is (= '(1 2 3) (take 4 '(1 2 3))))
  (is (= '(1 2 3) (take 4 [1 2 3]))))

(deftest drop
  (is (= '() (drop 0 nil)))
  (is (= '(1 2 3) (drop 0 '(1 2 3))))
  (is (= '(1 2 3) (drop 0 [1 2 3])))
  (is (= '(2 3) (drop 1 '(1 2 3))))
  (is (= '(2 3) (drop 1 [1 2 3])))
  (is (= '(3) (drop 2 '(1 2 3))))
  (is (= '(3) (drop 2 [1 2 3])))
  (is (= '() (drop 3 '(1 2 3))))
  (is (= '() (drop 3 [1 2 3])))
  (is (= '() (drop 4 '(1 2 3))))
  (is (= '() (drop 4 [1 2 3])))
  (is (= '(9) (drop 9 (range* 10)))))

(deftest partition
  (is (= '() (partition nil nil)))
  (is (= '() (partition 2 nil)))
  (is (= '() (partition 2 '(1))))
  (is (= '((1 2)) (partition 2 '(1 2))))
  (is (= '((1 2)) (partition 2 '(1 2 3))))
  (is (= '((1 2) (3 4)) (partition 2 [1 2 3 4])))
  (is (= '((1 2 3) (4 5 6)) (partition 3 [1 2 3 4 5 6 7 8]))))

(deftest eval
  (is (= 3 (eval '(+ 1 2))))
  (is (= "core-test" (eval '@*ns*)))
  (is (= 47 (eval '(+ FORTY-TWO 5)))))

(deftest python-builtin
  (is (= "42" (python/str 42)))
  (is (= 1234 (python/abs -1234)))
  (is (= 5 (python/max 1 5 3 2 4)))
  (is (= 1024 (python/pow 2 10))))

(deftest str
  (is (= "" (str)))
  (is (= "42" (str 42)))
  (is (= "42:hello" (str 42 :hello)))
  (is (= "42:helloworld" (str 42 :hello "world")))
  (is (= "hello:world" (str "hello" nil :world)))
  (is (= "" (str nil)))
  (is (= "" (str nil nil)))
  (is (= "foo" (str nil 'foo nil)))
  (is (= "[1 2 3]" (str [1 2 3])))
  (is (= "(1 2 3)" (str '(1 2 3))))
  (is (= "{:a 1}" (str {:a 1}))))

(deftest instance?
  (is (= true (instance? python/int 42)))
  (is (= false (instance? python/int 42.0)))
  (is (= true (instance? python/float 42.0)))
  (is (= false (instance? python/float 42)))
  (is (= true (instance? python/str "hello")))
  (is (= false (instance? python/str 42))))

(deftest re-find
  (is (= nil (re-find #"a" "hello")))
  (is (= "l" (re-find #"l" "hello")))
  (is (= '("ello" "el" "lo") (re-find #"(e.)(l.)" "hello"))))

(deftest mapcat
  (is (= '() (mapcat nil nil)))
  (is (= '() (mapcat list nil)))
  (is (= '(1 2 3) (mapcat list [1 2 3])))
  (is (= '(1 2 2 3 3 4) (mapcat (fn [x] [x (+ x 1)]) [1 2 3]))))

(deftest defrecord
  (let [Foo (eval '(defrecord Foo [a b c]))]
    (is (inspect/isclass Foo))
    (is (instance? Foo (Foo 1 2 3)))
    (is (= 1 (.-a (Foo 1 2 3))))
    (is (= 2 (.-b (Foo 1 2 3))))
    (is (= 3 (.-c (Foo 1 2 3))))
    (is (= 1 (:a (Foo 1 2 3))))
    (is (= 2 (:b (Foo 1 2 3))))
    (is (= 3 (:c (Foo 1 2 3))))
    (is (nil? (:d (Foo 1 2 3))))
    (is (= (Foo 1 2 3) (Foo 1 2 3)))
    (is (not= (Foo 1 2 3) (Foo 1 4 3)))
    (is (= (Foo 4 2 3) (assoc (Foo 1 2 3) :a 4)))
    (is (= (Foo 1 5 3) (assoc (Foo 1 2 3) :b 5)))
    (is (= (Foo 1 2 6) (assoc (Foo 1 2 3) :c 6)))))

(deftest doseq
  (is (= nil (eval '(let [sum (atom 0)]
                      (doseq [x [1 2 3]]
                        (.swap sum #(+ % x)))))))
  (is (= 6 (eval '(let [sum (atom 0)]
                    (doseq [x [1 2 3]]
                      (.swap sum #(+ % x)))
                    @sum))))
  (is (= 0 (eval '(let [sum (atom 0)]
                    (doseq [x []]
                      (.swap sum #(+ % x)))
                    @sum)))))
