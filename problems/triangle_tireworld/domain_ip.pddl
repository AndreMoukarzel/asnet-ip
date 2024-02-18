(define (domain triangle-tire)
  (:requirements :typing :strips :equality :probabilistic-effects :rewards)
  (:types location)
  (:predicates (vehicle-at ?loc - location)
	       (spare-in ?loc - location)
	       (road ?from - location ?to - location)
	       (not-flattire) (hasspare))
  (:action move-car
    :parameters (?from - location ?to - location)
    :precondition (and (vehicle-at ?from) (road ?from ?to) (not-flattire) (not (equal ?from ?to)))
    :effect 
		 (imprecise (0.0 .5) (and (vehicle-at ?to) (not (vehicle-at ?from)) (not (not-flattire)))
                (1.0 1.0) (and (vehicle-at ?to) (not (vehicle-at ?from))))
  )
  (:action loadtire
    :parameters (?loc - location)
    :precondition (and (vehicle-at ?loc) (spare-in ?loc))
    :effect (and (hasspare) (not (spare-in ?loc))))
  (:action changetire
    :precondition (hasspare)
    :effect (and (not (hasspare)) (not-flattire)))

)
