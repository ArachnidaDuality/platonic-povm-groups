import LeanExactness.AppendixD1
import LeanExactness.Solids
import LeanExactness.Weights
import LeanExactness.Sanity

/-! Self-check: every headline result must rest only on Lean's three standard axioms
(`propext`, `Classical.choice`, `Quot.sound`) — in particular never on `sorryAx`. -/

-- Lemma 5, the rotation-proof witness
#print axioms AppendixD1.det_mem_of_pose
#print axioms AppendixD1.witness_mem_of_exists_pose
#print axioms AppendixD1.gram_det_eq_witness_sq
#print axioms AppendixD1.gram_det_change_of_basis

-- Lemma 5's last clause: any independent triple decides
#print axioms AppendixD1.inv_entries_mem
#print axioms AppendixD1.det_eq_mul_of_gram_mem
#print axioms AppendixD1.det_mem_iff_of_gram_mem
#print axioms AppendixD1.witness_mem_iff_of_vertex_inner_mem
#print axioms AppendixD1.witness_notMem_of_witness_notMem

-- Lemma 6, membership in K_R
#print axioms AppendixD1.sqrt3_notMem_KR
#print axioms AppendixD1.sqrt_notMem_KR
#print axioms AppendixD1.sqrt3_notMem_KR'
#print axioms AppendixD1.sqrt6_notMem_KR
#print axioms AppendixD1.sqrt7_notMem_KR
#print axioms AppendixD1.sqrt15_notMem_KR
#print axioms AppendixD1.sqrt30_notMem_KR
#print axioms AppendixD1.eq_of_squarefree_of_isSquare_mul
#print axioms AppendixD1.sqrt_mem_KR_iff_of_squarefree
#print axioms AppendixD1.descent
#print axioms AppendixD1.icosahedral_witness_not_sq
#print axioms AppendixD1.icosahedral_normalizer_not_sq

-- K_R as a field, and the coset forms the solids actually need
#print axioms AppendixD1.KRsubfield
#print axioms AppendixD1.mul_notMem_KR
#print axioms AppendixD1.div_notMem_KR
#print axioms AppendixD1.smul_sqrt3_notMem_KR
#print axioms AppendixD1.sqrt_icosahedral_normalizer_notMem_KR

-- Lemmas 5 and 6 composed
#print axioms AppendixD1.no_pose_of_witness_mem_sqrt3_coset
#print axioms AppendixD1.no_pose_of_gram_eq_icosahedral

-- the four inexact solids, at their atlas coordinates
#print axioms AppendixD1.Solids.witness_tetU
#print axioms AppendixD1.Solids.witness_ico_sq
#print axioms AppendixD1.Solids.no_pose_tetrahedron
#print axioms AppendixD1.Solids.no_pose_cube
#print axioms AppendixD1.Solids.no_pose_dodecahedron
#print axioms AppendixD1.Solids.no_pose_icosahedron

-- Corollary 7's arithmetic, as a statement about the radius
#print axioms AppendixD1.Solids.vertex_notMem_of_normalizer
#print axioms AppendixD1.Solids.no_alignment_sqrt3
#print axioms AppendixD1.Solids.no_alignment_ico

-- Theorem 4's arithmetic core: the weight obstruction
#print axioms AppendixD1.Weights.Rring_le_halvedIntegral
#print axioms AppendixD1.Weights.isDyadic_of_rat_mem_Rring
#print axioms AppendixD1.Weights.rat_mem_Rring_of_isDyadic
#print axioms AppendixD1.Weights.isDyadic_two_div_iff
#print axioms AppendixD1.Weights.weight_obstruction
#print axioms AppendixD1.Weights.octahedron_weight_notMem
#print axioms AppendixD1.Weights.icosahedron_weight_notMem
#print axioms AppendixD1.Weights.dodecahedron_weight_notMem
#print axioms AppendixD1.Weights.tetrahedron_weight_mem
#print axioms AppendixD1.Weights.cube_weight_mem

-- the ring and the field, identified: Frac R = K and Frac R n R = K_R
#print axioms AppendixD1.Weights.Rring_le_KCsubfield
#print axioms AppendixD1.Weights.ofReal_mem_fracRring
#print axioms AppendixD1.Weights.fracRring_eq_KC
#print axioms AppendixD1.Weights.mem_fracRring_iff_KR

-- the sanity controls that show the classification and Lemma 5's last clause are not vacuous:
-- the tetrahedron's second triple, derived rather than recomputed
#print axioms AppendixD1.Sanity.tetR_inner_mem
#print axioms AppendixD1.Sanity.witness_tetR_012_notMem
#print axioms AppendixD1.Sanity.witness_tetR_032_notMem
