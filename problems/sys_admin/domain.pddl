;; System Administrator Problem, variant of (Guestrin, Koller, Parr; IJCAI-2001)
;; Encoded in PPDDL by Scott Sanner with assistance of Dan Bryce & Olivier Buffet.
;;
;; Note: The original SysAdmin is discounted infinite horizon, with an additive
;; reward function, and a transition function probability that scales
;; according to the number of connected computers that are "up".
;; The latter two additive aspects cannot be encoded in a lifted manner
;; in PPDDL.
;;
;; Here, a computer may fails if at least one of its upstream
;; connections has failed, so it is important to reboot the computers
;; with the highest downstream impact first.
(define (domain sysadmin)
(:requirements :typing :equality :probabilistic-effects :negative-preconditions :sysadmin :rewards)
(:types comp)
(:predicates (up ?c) (conn ?c ?d))

;; Don’t need for finite horizon problems
;;(:action noop)

(:action reboot
 :parameters (?x - comp)
 :effect (probabilistic 0.9 (up ?x)
		(forall (?d - comp)
			(probabilistic
				0.6 (when (exists (?c - comp) (and (conn ?c ?d)
													(not (up ?c))
													(not (= ?x ?d))))
						(not (up ?d))
)))))
)
