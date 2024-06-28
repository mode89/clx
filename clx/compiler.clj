(in-ns 'clx.compiler)

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
             token (Token (.group m 2) line)]
         (cons token (tokenize (next matches) line)))))))

(declare read-form)

(defn read-string [s]
  (first (read-form (tokenize s))))

(declare read-atom)

(defn read-form [tokens]
  (let [token (first tokens)
        rtokens (rest tokens)]
    [(read-atom (.-text token)) rtokens]))

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
