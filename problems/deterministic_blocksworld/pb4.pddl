(define (problem pb4)
  (:domain blocksworld)
  (:objects a b c d - block)
  (:init (onTable a) (on b a) (on c b) (on d c) (clear d))
  (:goal (and (on b a) (on c b) (on a d))))