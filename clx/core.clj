(in-ns clx.core)

(import* operator)

(def throw
  (fn* clx.core/throw [x]
    (python* "raise " x)))

(def Exception
  (python* "Exception"))

(def if ^{:macro? true}
  (fn* clx.core/if [pred then else]
    `(cond
       ~pred ~then
       true ~else)))

(def if-not ^{:macro? true}
  (fn* clx.core/if-not [pred then else]
    `(cond
       ~pred ~else
       true ~then)))

(def when ^{:macro? true}
  (fn* clx.core/when [pred & body]
    `(cond ~pred (do ~@body))))

(def when-not ^{:macro? true}
  (fn* clx.core/when-not [pred & body]
    `(cond
       ~pred nil
       true (do ~@body))))

(def assert ^{:macro? true}
  (fn* clx.core/assert [tst & msg]
    `(when-not ~tst
       (throw (Exception ~@msg)))))

(def fn ^{:macro? true}
  (fn* clx.core/fn [& args]
    (let* [arg1 (first args)]
      (if (symbol? arg1)
        `(fn* ~arg1 ~(second args)
          (do ~@(rest (rest args))))
        `(fn* ~arg1
          (do ~@(rest args)))))))

(def defn ^{:macro? true}
  (fn* clx.core/defn [name params & body]
    `(def ~name
      (with-meta
        (fn ~name ~params ~@body)
        ~(meta name)))))

(def defmacro ^{:macro? true}
  (fn* clx.core/defmacro [name params & body]
    `(defn ~(with-meta name {:macro? true}) ~params
      (do ~@body))))

(defn even? [x]
  (python* "not " x " & 1"))

(defn odd? [x]
  (python* x " & 1 == 1"))

(defn inc [x]
  (python* x " + 1"))

(defn dec [x]
  (python* x " - 1"))

(defmacro let [bindings & body]
  (assert (vector? bindings) "bindings must be a vector")
  (assert (even? (count bindings))
    "bindings must have an even number of elements")
  `(let* ~bindings
    (do ~@body)))

(defmacro when-let [bindings & body]
  (assert (vector? bindings) "bindings must be a vector")
  (assert (operator/eq 2 (count bindings))
    "bindings must have exactly two elements")
  (let [bname (bindings 0)
        bvalue (bindings 1)]
    `(let [~bname ~bvalue]
       (when ~bname ~@body))))

(defn name [x]
  (python* x ".name"))

(defmacro set! [obj field value]
  `(let [obj# ~obj
         value# ~value]
     (python* obj# "." ~(name field) " = " value#)))

(defmacro lazy-seq [& body]
  `(lazy-seq* (fn [] ~@body)))
