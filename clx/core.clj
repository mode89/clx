(in-ns 'clx.core)

(import* operator)

(def throw
  (fn* clx.core/throw [x]
    (python* "raise " x)))

(def Exception
  (python* "Exception"))

(def ^{:macro? true} if
  (fn* clx.core/if [pred then else]
    `(cond
       ~pred ~then
       true ~else)))

(def ^{:macro? true} if-not
  (fn* clx.core/if-not [pred then else]
    `(cond
       ~pred ~else
       true ~then)))

(def ^{:macro? true} when
  (fn* clx.core/when [pred & body]
    `(cond ~pred (do ~@body))))

(def ^{:macro? true} when-not
  (fn* clx.core/when-not [pred & body]
    `(cond
       ~pred nil
       true (do ~@body))))

(def ^{:macro? true} assert
  (fn* clx.core/assert [tst & msg]
    `(when-not ~tst
       (throw (Exception ~@msg)))))

(def multi-arity-fn
  (fn* multi-arity-fn [fname decls]
    (let* [variadic? (fn* [args]
                       (loop* [args* args]
                         (when (seq args*)
                           (if (operator/eq '& (first args*))
                             true
                             (recur (rest args*))))))
           arity-name (fn* [arity]
                        (when fname
                          (vector
                            (symbol
                              (let* [fname* (python* "str(" fname ")")
                                     arity* (python* "str(" arity ")")]
                                (python*
                                  fname*
                                  " + \"-arity-\" + "
                                  arity*))))))
           decl-entry (fn* [decl]
                        (let* [args (first decl)
                               _ (assert (vector? args)
                                   "args must be a vector")
                               body (rest decl)
                               arity (if (variadic? args)
                                       :variadic
                                       (count args))]
                          `(list ~arity
                             (fn* ~@(arity-name arity) [~@args]
                               (do ~@body)))))]
      `(let* [arities# (list ~@(loop* [decls* decls
                                       entries (list)]
                                 (if (seq decls*)
                                   (recur
                                     (rest decls*)
                                     (cons (decl-entry (first decls*))
                                           entries))
                                   entries)))
              arities-dict# (python* "dict(" arities# ")")
              variadic# (.get arities-dict# :variadic)]
         (fn* ~@(when fname [fname]) [& args]
           (python*
             "f = " arities-dict# ".get(len(" args "), " variadic# ")\n"
             "if f is None:\n"
             "    raise Exception(\"Wrong number of arguments\")\n"
             "f(*" args ")"))))))

(def ^{:macro? true} fn
  (fn* clx.core/fn [& args]
    (if (symbol? (first args))
      (let* [fname (first args)]
        (if (vector? (second args))
          (let* [params (second args)]
            `(fn* ~fname ~params
               (do ~@(rest (rest args)))))
          (multi-arity-fn fname (rest args))))
      (if (vector? (first args))
        (let* [params (first args)]
          `(fn* ~params
             (do ~@(rest args))))
        (multi-arity-fn (gensym "___fn_") args)))))

(def ^{:macro? true} defn
  (fn* clx.core/defn [fname & decl]
    `(def ~fname
        (fn ~(symbol *ns* (.-name fname)) ~@decl))))

(def ^{:macro? true} defmacro
  (fn* clx.core/defmacro [name & decl]
    `(defn ~(with-meta name {:macro? true}) ~@decl)))

(defn nil? [x]
  (python* x " is None"))

(defn string? [x]
  (python* "type(" x ") is str"))

(defn some? [x]
  (python* x " is not None"))

(defn not [x]
  (if x false true))

(defn even? [x]
  (python* "not " x " & 1"))

(defn odd? [x]
  (python* x " & 1 == 1"))

(defn inc [x]
  (python* x " + 1"))

(defn dec [x]
  (python* x " - 1"))

(def + operator/add)
(def - operator/sub)
(def * operator/mul)
(def / operator/truediv)
(def = operator/eq)
(def not= operator/ne)
(def < operator/lt)
(def > operator/gt)
(def <= operator/le)
(def >= operator/ge)

(defmacro and
  ([] true)
  ([x] x)
  ([x & xs]
   `(let [x# ~x]
     (if x#
       (and ~@xs)
       x#))))

(defmacro or
  ([] nil)
  ([x] x)
  ([x & xs]
   `(let [x# ~x]
     (if x#
       x#
       (or ~@xs)))))

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

(defn map [f coll]
  (lazy-seq
    (when-let [s (seq coll)]
      (cons (f (first s)) (map f (rest s))))))

(defn filter [pred coll]
  (lazy-seq
    (when-let [s (seq coll)]
      (let [s0 (first s)]
        (if (pred s0)
          (cons s0 (filter pred (rest s)))
          (filter pred (rest s)))))))

(defn reduce
  ([f coll]
   (reduce f (first coll) (rest coll)))
  ([f init coll]
   (loop* [acc init
           coll (seq coll)]
     (if coll
       (recur (f acc (first coll)) (rest coll))
       acc))))

(load-file "clx/python.clj")

(defn str [& args]
  (.join ""
    (map (fn [arg]
           (if (some? arg)
             (pr-str arg)
             ""))
         args)))

(defn instance? [t x]
  (python* "isinstance(" x ", " t ")"))

(defn re-find [re s]
  (when-let [match (.search re s)]
    (let [full-match (python* match "[0]")
          groups (seq (.groups match))]
      (if (some? groups)
        (cons full-match groups)
        full-match))))
