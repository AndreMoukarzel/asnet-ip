(define (domain det-blocksworld)
  (:requirements :strips :negative-preconditions :equality)
  (:types block)
  (:predicates (holding ?b - block) (emptyhand) (on-table ?b - block) (on ?b1 ?b2 - block) (clear ?b - block))

  (:action pickup
    :parameters (?ob - block)
    :precondition (and (clear ?ob) (on-table ?ob) (emptyhand))
    :effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob)) (not (emptyhand)))
  )

  (:action putdown
    :parameters (?ob - block)
    :precondition (holding ?ob)
    :effect (and (clear ?ob) (on-table ?ob) (not (holding ?ob)) (emptyhand))
  )

  (:action stack
    :parameters (?ob ?underob - block)
    :precondition (and (clear ?underob) (holding ?ob) (not (equal ?ob ?underob)))
    :effect (and (clear ?ob) (on ?ob ?underob) (not (clear ?underob)) (not (holding ?ob)) (emptyhand))
  )

  (:action unstack
    :parameters (?ob ?underob - block)
    :precondition (and (on ?ob ?underob) (clear ?ob) (not (equal ?ob ?underob)) (emptyhand))
    :effect (and (holding ?ob) (clear ?underob) (not (on ?ob ?underob)) (not (clear ?ob)) (not (emptyhand)))
  )
)