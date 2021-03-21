

\version "2.20.0"  % necessary for upgrading to future LilyPond versions.

\header{
  title = "A scale in LilyPond"
  subtitle = "For more information on using LilyPond, please see
http://lilypond.org/introduction.html"
}

\relative {

	\tempo 4=130
  	e16 f g a b a g f e r4 r8 r16 
  	r1
	e16 f g a b a g f e r4 r8 r16 
  	r1
	e16 f g a b a g f e f g a b a g f 
	e f g a b a g f e r4 r8 r16
	


}
