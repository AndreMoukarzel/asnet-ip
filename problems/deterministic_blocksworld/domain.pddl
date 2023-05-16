(define (domain blocksworld)
  (:requirements :strips :negative-preconditions)
  (:predicates (clear ?x - block) (onTable ?x - block) (holding ?x - block) (on ?x ?y - block))

  (:action pickup
    :parameters (?ob - block)
    :precondition (and (clear ?ob) (onTable ?ob))
    :effect (and (holding ?ob) (not (clear ?ob)) (not (onTable ?ob)))
  )

  (:action putdown
    :parameters (?ob - block)
    :precondition (holding ?ob)
    :effect (and (clear ?ob) (onTable ?ob) (not (holding ?ob)))
  )

  (:action stack
    :parameters (?ob ?underob - block)
    :precondition (and (clear ?underob) (holding ?ob) (not (equal ?ob ?underob)))
    :effect (and (clear ?ob) (on ?ob ?underob) (not (clear ?underob)) (not (holding ?ob)))
  )

  (:action unstack
    :parameters (?ob ?underob - block)
    :precondition (and (on ?ob ?underob) (clear ?ob) (not (equal ?ob ?underob)))
    :effect (and (holding ?ob) (clear ?underob) (not (on ?ob ?underob)) (not (clear ?ob)))
  )
)