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
        (fn ~(symbol @*ns* (.-name fname)) ~@decl))))

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

(defn zero? [x]
  (python* x " == 0"))

(defn pos? [x]
  (python* x " > 0"))

(defn neg? [x]
  (python* x " < 0"))

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
    (map #(if (some? %)
             (pr-str %)
             "")
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

(defn mapcat [f coll]
  (concat* (map f coll)))

; TODO it should update namespaces without touching py_globals
(defmacro declare [name]
  `(def ~name nil))

(defmacro type* [cname cbases & fspecs]
  `(python/type
     ~(str cname)
     (python/tuple ~cbases)
     (python/dict
       (hash-map
         ~@(mapcat
             (fn [fspec]
               (let [fname (first fspec)
                     fname* (symbol (str cname "." fname))
                     fargs (second fspec)
                     fbody (rest (rest fspec))]
                 [(str fname)
                  `(fn ~fname* ~fargs
                     ~@fbody)]))
             fspecs)))))

(defmacro defrecord [tname fields]
  (assert (symbol? tname) "record name must be a symbol")
  (assert (vector? fields) "record fields must be a vector")
  (let [qtname (symbol @*ns* (name tname))
        self (gensym "self_")
        other (gensym "other_")
        make-assoc (fn make-assoc [field]
                     (let [value (gensym "value_")
                           fname (symbol (str qtname "-assoc-" field))]
                       `(fn ~fname [~self ~value]
                          (~tname
                            ~@(map
                                #(if (= % field)
                                    value
                                    `(. ~self ~(symbol (str "-" %))))
                                fields)))))]
    `(do (declare ~tname)
         (let [assocs# (python/dict
                         (hash-map
                           ~@(mapcat
                               #(vector (keyword %) (make-assoc %))
                               fields)))]
          (def ~tname
            (clx.core/type* ~qtname [clx.core/IAssociative]
              (__init__ [~self ~@fields]
                ~@(map (fn [f]
                         `(python* ~self "." ~f " = " ~f))
                       fields))
              (__eq__ [~self ~other]
                (and (instance? ~tname ~other)
                     ~@(map #(let [f* (symbol (str "-" %))]
                              `(= (. ~self ~f*) (. ~other ~f*)))
                            fields)))
              (lookup [~self k not-found]
                (python/getattr ~self (.-munged k) not-found))
              (assoc [~self & kvs#]
                (python*
                  "obj = " ~self "\n"
                  "for k, v in zip(" kvs# "[::2], " kvs# "[1::2]):\n"
                  "  obj = " assocs# "[k](obj, v)\n"
                  "obj"))))
          ~tname))))

(defn take [n coll]
  (lazy-seq
    (when (pos? n)
      (when-let [s (seq coll)]
        (cons (first s) (take (dec n) (rest s)))))))

(defn drop [n coll]
  (lazy-seq
    (loop* [n n
            coll (seq coll)]
      (if (and (pos? n) coll)
        (recur (dec n) (rest coll))
        coll))))

(defn partition [n coll]
  (lazy-seq
    (when-let [s (seq coll)]
      (let [s0 (take n s)]
        (when (= n (count s0))
          (cons s0 (partition n (drop n s))))))))

(defmacro doseq [bindings & body]
  (assert (vector? bindings) "bindings must be a vector")
  ; TODO allow multiple bindings
  (assert (= 2 (count bindings)) "bindings must have exactly two elements")
  (let [b (bindings 0)
        s (bindings 1)]
    `(loop* [s# (seq ~s)]
      (when s#
        (let [~b (first s#)]
          ~@body
          (recur (next s#)))))))
