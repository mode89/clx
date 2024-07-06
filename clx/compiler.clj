(in-ns 'clx.compiler)

(import* ast)

(defrecord Token [text line])

(def RE-TOKEN
  (re-pattern
    (str
      "([\\s,]*)" ; whitespaces
      "("
        "~@" "|" ; unquote-splicing
        "[\\[\\]{}()'`~^@]" "|" ; special characters
        "\"(?:[\\\\].|[^\\\"])*\"?" "|" ; strings
        ";.*" "|" ; comments
        "[^\\s\\[\\]{}()'\"`@,;]+" ; symbols
      ")")))

(defn tokenize
  ([text]
   (tokenize (seq (.finditer RE-TOKEN text)) 1))
  ([matches line]
   (lazy-seq
     (when matches
       (let [m (first matches)
             spaces (.group m 1)
             token (Token. (.group m 2) line)]
         (cons token (tokenize (next matches) line)))))))

(declare read-form)

(defn read-string [s]
  (first (read-form (tokenize s))))

(declare read-collection)
(declare read-atom)

(defn read-form [tokens]
  (let [t (.-text (first tokens))
        ts (rest tokens)]
    (cond
      (= "(" t) (read-collection ts list ")")
      (= "[" t) (read-collection ts vector "]")
      (= "{" t) (read-collection ts hash-map "}")
      (= "'" t) (let [[form ts*] (read-form ts)]
                  [(list 'quote form) ts*])
      :else [(read-atom t) ts])))

(declare read-string-literal)

(defn read-atom [token]
  (assert (string? token))
  (cond
    (re-find #"^\d+$" token) (python/int token)
    (re-find #"^\d+\.\d+$" token) (python/float token)
    (re-find #"^\"(?:[\\].|[^\\\"])*\"$" token) (read-string-literal token)
    (= "\"" (first token)) (throw (Exception "Unterminated string"))
    (= "true" token) true
    (= "false" token) false
    (= "nil" token) nil
    (= ":" (first token)) (keyword (subs token 1))
    :else (symbol token)))

(defn read-string-literal [token]
  (-> (subs token 1 (dec (count token)))
      (. replace "\\\"" "\"")
      (. replace "\\\\" "\\")))

(defn read-collection [ts ctor end]
  (loop [elements []
         ts ts]
    (let [token (first ts)]
      (assert (some? token) (str "Expected '" end "'"))
      (assert (instance? Token token))
      (if (= end (.-text token))
        [(apply ctor elements) (rest ts)]
        (let [[item ts*] (read-form ts)]
          (recur (conj elements item) ts*))))))

(def MUNGE-TABLE
  {"-" "_DASH_"
   "." "_DOT_"
   ":" "_COLON_"
   "+" "_PLUS_"
   "*" "_STAR_"
   "&" "_AMPER_"
   ">" "_GT_"
   "<" "_LT_"
   "=" "_EQ_"
   "%" "_PERCENT_"
   "#" "_SHARP_"
   "!" "_BANG_"
   "?" "_QMARK_"
   "'" "_SQUOTE_"
   "|" "_BAR_"
   "/" "_SLASH_"
   "$" "_DOLLAR_"})

(def SPECIAL-NAMES
  (hash-set
    "and"
    "as"
    "assert"
    "async"
    "await"
    "break"
    "class"
    "continue"
    "def"
    "del"
    "elif"
    "else"
    "except"
    "False"
    "finally"
    "for"
    "from"
    "global"
    "if"
    "import"
    "in"
    "is"
    "lambda"
    "None"
    "or"
    "return"
    "True"
    "try"
    "while"
    "with"
    "yield"))

(defn munge [obj]
  (let [name (cond
               (or (symbol? obj)
                   (keyword? obj)) (if-some [ns (namespace obj)]
                                     (str ns "/" (name obj))
                                     (name obj))
               (string? obj) obj
               :else (throw (Exception.
                              (str "'munge' expects a symbol, "
                                   "keyword or a string"))))]
    (assert (not (and (= "_" (last name))
                      (contains? SPECIAL-NAMES (python* @name "[:-1]"))))
      (str "name '" name "' is reserved"))
    (.join ""
      (map #(get MUNGE-TABLE % %)
           (if (contains? SPECIAL-NAMES name)
             (str name "_")
             name)))))

(defrecord LocalContext [env loop-bindings tail? top-level? line column])

(defn default-local-context []
  (LocalContext. {} nil false true 1 1))

(defrecord Context [local namespaces py-globals])

(declare intern*)

(defn make-context []
  (let [ctx (Context.
              (Var. (default-local-context))
              (atom {})
              (python/dict))]
    (doto ctx
      (intern*))))

(declare compile*)

(defn eval-form [ctx form]
  (let [[result body] (compile* ctx form)]
    (python/exec
      (python/compile
        (apply-kw* ast/Module body {"type_ignores" []})
        "<no-file>" "exec")
      (:py-globals ctx))
    (python/eval
      (python/compile
        (apply-kw* ast/Expression result {"type_ignores" []})
        "<no-file>" "eval")
      (:py-globals ctx))))

(declare compile-def)

(defn compile* [ctx form]
  (cond
    (list? form)
    (let [head (first form)]
      (cond
        (= 'def head) (compile-def ctx form)))))

(defn compile-def [ctx form]
  (assert (= 3 (count form)) "'def' form expects 2 arguments")
  (let [name (second form)]))
