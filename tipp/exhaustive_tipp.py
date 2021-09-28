"""
Created on Sep 27, 2021

@author: Gillian Chu
"""
import sys
import random
import argparse
import os
import shutil
from sepp import get_logger
from sepp.alignment import MutableAlignment, ExtendedAlignment, _write_fasta
from sepp.exhaustive import JoinAlignJobs, ExhaustiveAlgorithm 
from sepp.exhaustive import get_placement_job_name, AbstractAlgorithm # TIPP
from sepp.jobs import PastaAlignJob
from sepp.jobs import HMMBuildJob, HMMSearchJob, HMMAlignJob, PplacerJob, ExternalSeppJob # TIPP
from sepp.filemgr import get_temp_file
from sepp.config import options, valid_decomp_strategy
import sepp.config
from sepp.math_utils import lcm
from sepp.problem import SeppProblem, RootProblem # TIPP
from sepp.scheduler import JobPool, Join # TIPP
from sepp.tree import PhylogeneticTree # TIPP
from dendropy.datamodel.treemodel import Tree # TIPP
import dendropy # TIPP
import pickle # TIPP
from functools import reduce
from sepp.backtranslate import backtranslate

_LOG = get_logger(__name__)


# UPPJoinAlignJobs inherits from JoinAlignJobs which inherits from Join
# TIPPJoinSearchJobs inherits from Join

# TIPP has TIPPJoinAlignJobs(JoinAlignJobs)
# TIPP also has TIPPJoinSearchJobs(Join)

# UPP has UPPJoinAlignJobs(JoinAlignJobs)
# UPP inherits JoinSearchJobs(Join) from ExhaustiveAlgorithm

### TIPP difference
class TIPPJoinSearchJobs(Join): 
    """
    After all search jobs have finished on tips, we need to figure out which 
    fragment goes to which subset and start aligning fragments.
    This join takes care of that step.
    """
    def __init__(self, alignment_threshold):
        Join.__init__(self)
        self.alignment_threshold = alignment_threshold
        self.root_problem = None
    
    def setup_with_root_problem(self, root_problem):
        self.root_problem = root_problem
        for p in root_problem.iter_leaves():
            self.add_job(p.jobs['hmmsearch'])

    def figureout_fragment_subset(self):
        ''' Figure out which fragment should go to which subproblem '''
        if "fragments.distribution.done" in self.root_problem.annotations:
            return

        ### TIPP difference: keeping bitscore instead of e-val
        # actually exqual to exhaustive otherwise, although syntax differences
        bitscores = dict([(name, []) for name in list(
            self.root_problem.fragments.keys())])
        for fragment_chunk_problem in self.root_problem.iter_leaves():
            align_problem = fragment_chunk_problem.get_parent()
            assert isinstance(align_problem, SeppProblem)
            '''  For each subproblem start with an empty set of fragments, 
            and add to them as we encounter new best hits for that subproblem '''
            if align_problem.fragments is None:
                align_problem.fragments = MutableAlignment()
            search_res = fragment_chunk_problem.get_job_result_by_name("hmmsearch")
            for key in list(search_res.keys()):
                '''keep a list of all hits, and their bit scores'''
                bitscores[key].append((search_res[key][1], align_problem))
        for frag, tuplelist in bitscores.items():
            _LOG.warning("Fragment %s is not scored against any subset" % str(frag))
            if len(tuplelist) == 0:
                _LOG.warning("Fragment %s is not scored against any subset" % str(frag))
                continue
            '''convert bitscores to probabilities'''
            denum = sum(math.pow(2, min(x[0], 1022)) for x in tuplelist)
            tuplelist = [
                ((math.pow(2, min(x[0], 1022)) / denum * 1000000), x[1])
                for x in tuplelist]
            ''' sort subsets by their probability '''
            tuplelist.sort(reverse=True, key=lambda x: x[0])
            ''' Find enough subsets to reach the threshold '''
            selected = tuplelist[
                0: max(1, reduce(
                    lambda x, y: (x[0], None)
                    if x[1] is None else
                    (y[0], x[1] + y[1]) if x[1] < int(
                        1000000 * self.alignment_threshold) else 
                    (y[0], None),
                    enumerate([x[0] for x in tuplelist]))[0])]
            ''' Renormalized the selected list to add up to 1 '''
            renorm = 0
            for (prob, align_problem) in selected:
                renorm = renorm + prob / 100000
            renorm = 1 / renorm

            _LOG.debug("Fragment %s assigned to %d subsets" % (frag, len(selected)))
            ''' Rename the fragment and assign it to the respective subsets '''
            for (prob, align_problem) in selected:
                postfix = prob * renorm if \
                    options().exhaustive.weight_placement_by_alignment.\
                    lower() == "true" \
                    else 1000000
                frag_rename = "%s_%s_%d" % (frag, align_problem.label, postfix)
                align_problem.fragments[frag_rename] = \
                    self.root_problem.fragments[frag]
        self.root_problem.annotations["fragments.distribution.done"] = 1

    # explicitly copied from JoinSearchJobs
    def perform(self):
        self.figureout_fragment_subset()

        # for each alignment subproblem,
        # 1) make sure its fragments are evenly distributed to fragment chunks.
        # 2) setup alignment jobs for its children and enqueue them
        alg_problems = [alg
                        for p in self.root_problem.children
                        for alg in p.children]
        for alg_problem in alg_problems:
            assert isinstance(alg_problem, SeppProblem)
            chunks = len(alg_problem.get_children())
            fragment_chunks = alg_problem.fragments.divide_to_equal_chunks(chunks)

            # now setup alignment jobs and enqueue them
            for (i, fragment_chunk_problem) in enumerate(alg_problem.children):
                fragment_chunk_problem.fragments = fragment_chunks[i]
                aj = fragment_chunk_problem.jobs['hmmalign']
                assert isinstance(aj, HMMAlignJob)
                # first complete setting up alignments"
                aj.hmmmodel = alg_problem.get_job_result_by_name('hmmbuild')
                aj.base_alignment = alg_problem.jobs["hmmbuild"].infile

                if fragment_chunk_problem.fragments is None \
                    or fragment_chunk_problem.fragments.is_empty():
                    aj.fake_run = True
                else:
                    fragment_chunk_problem.fragments.write_to_path(aj.fagmetns)
                ''' Now the align job can be put on the queue. '''
                JobPool().enqueue_job(aj)

    def __str__(self):
        return "join search jobs for all tips of ", self.root_problem



class TIPPJoinAlignJobs(JoinAlignJobs):
    """
    After all alignments jobs for a placement subset have finished,
    we need to build those extended alignments. This join takes care of that
    step.
    """

    def __init__(self, placer):
        JoinAlignJobs.__init__(self)
        self.placer = placer

    def perform(self):
        pp = self.placement_problem
        fullExtendedAlignments = self.merge_subalignments()

        # from UPP
        assert isinstance(pp, SeppProblem)
        pp.annotations["search_join_object"] = self
        
        for i in range(0, self.root_problem.fragment_chunks):
            fullExtendedAlignment = fullExtendedAlignments[i]
            queryExtendedAlignment = \
                fullExtendedAlignment.get_fragments_readonly_alignment()
            base_alignment = fullExtendedAlignment.get_base_readonly_alignment()
            pj = pp.jobs[get_placement_job_name(i)]

            if queryExtendedAlignment.is_empty():
                pj.fake_run = True

            if self.placer == "pplacer":
                assert isinstance(pj, PplacerJob)
                queryExtendedAlignment.write_to_path(
                    pj.extended_alignment_file)
                base_alignment.write_to_path(pj.backbone_alignment_file)
            output = open(pj.full_extended_alignment_file, 'wb')
            pickle.dump(fullExtendedAlignment, output)
            output.close()

            JobPool().enqueue_job(pj)
        
    def __str__(self):
        return "join align jobs for tips of ", self.placement_problem

# This is the class of interest when run() is called, inherits from AbstractAlg
# differences are to unpickle large input file, use our own get_merge_job,  add
# the alignment_threshold, load reference and add threshold check before 
# connect_jobs(). 
class TIPPExhaustiveAlgorithm(ExhaustiveAlgorithm):
    """
    This implements the exhaustive algorithm where all alignments subsets
    are searched for every fragment. This is for UPP, meaning that no placement
    is performed, and that there is always only one placement subset
    (currently).
    """

    def __init__(self):
        ExhaustiveAlgorithm.__init__(self)
        self.pasta_only = False
        self.filtered_taxa = []

        self.alignment_threshold = self.options.alignment_threshold
        self.placer = self.options.exhaustive.placer.lower()
        self.push_down = True if self.options.push_down is True else False
        _LOG.info("Will push fragments %s from their placement edge." % ("down" if self.push_down else "up"))

    def generate_backbone(self):
        _LOG.info("Reading input sequences: %s" % self.options.sequence_file)
        sequences = MutableAlignment()
        sequences.read_file_object(self.options.sequence_file)
        sequences.degap()
        fragments = MutableAlignment()
        if options().median_full_length is not None \
                or options().full_length_range is not None:
            if options().median_full_length == -1:
                seq_lengths = sorted(
                    [len(seq) for seq in list(sequences.values())])
                lengths = len(seq_lengths)
                l2 = int(lengths / 2)
                if lengths % 2:
                    options().median_full_length = \
                        (seq_lengths[l2] + seq_lengths[l2 + 1]) / 2.0
                else:
                    options().median_full_length = seq_lengths[l2]
            if options().full_length_range is not None:
                L = sorted(int(x) for x in options().full_length_range.split())
                min_length = L[0]
                max_length = L[1]
            else:
                (min_length, max_length) = (
                    int(options().median_full_length * (
                            1 - options().backbone_threshold)),
                    int(options().median_full_length * (
                            1 + options().backbone_threshold)))
            _LOG.info("Full length sequences are set to be from "
                      "%d to %d character long" % (min_length, max_length))
            frag_names = [name for name in sequences
                          if len(sequences[name]) > max_length or
                          len(sequences[name]) < min_length]
            if len(frag_names) > 0:
                _LOG.info(
                    "Detected %d fragmentary sequences" % len(frag_names))
                fragments = sequences.get_hard_sub_alignment(frag_names)
                [sequences.pop(i) for i in list(fragments.keys())]
        if options().backbone_size is None:
            options().backbone_size = min(1000, int(sequences.get_num_taxa()))
            _LOG.info("Backbone size set to: %d" % options().backbone_size)
        if options().backbone_size > len(list(sequences.keys())):
            options().backbone_size = len(list(sequences.keys()))
        sample = sorted(random.sample(
            sorted(list(sequences.keys())), options().backbone_size))
        backbone_sequences = sequences.get_hard_sub_alignment(sample)
        _LOG.debug("Backbone: %s" % (sorted(list(backbone_sequences.keys()))))
        [sequences.pop(i) for i in list(backbone_sequences.keys())]

        _LOG.info("Writing backbone set. ")
        backbone = get_temp_file("backbone", "backbone", ".fas")
        _write_fasta(backbone_sequences, backbone)

        _LOG.info("Generating pasta backbone alignment and tree. ")
        pastaalign_job = PastaAlignJob()
        molecule_type = options().molecule
        if options().molecule == 'amino':
            molecule_type = 'protein'
        pastaalign_job.setup(backbone, options().backbone_size,
                             molecule_type, options().cpu,
                             **vars(options().pasta))
        pastaalign_job.run()
        (a_file, t_file) = pastaalign_job.read_results()

        shutil.copyfile(t_file, self.get_output_filename("pasta.fasttree"))
        shutil.copyfile(a_file, self.get_output_filename("pasta.fasta"))

        options().placement_size = self.options.backbone_size
        options().alignment_file = open(
            self.get_output_filename("pasta.fasta"))
        options().tree_file = open(self.get_output_filename("pasta.fasttree"))
        _LOG.info(
            "Backbone alignment written to %s.\nBackbone tree written to %s"
            % (options().alignment_file, options().tree_file))
        sequences.set_alignment(fragments)
        if len(sequences) == 0:
            sequences = MutableAlignment()
            sequences.read_file_object(open(self.options.alignment_file.name))
            self.results = ExtendedAlignment(fragment_names=[])
            self.results.set_alignment(sequences)
            _LOG.info(
                "No query sequences to align.  Final alignment saved as %s"
                % self.get_output_filename("alignment.fasta"))
            self.output_results()
            sys.exit(0)
        else:
            query = get_temp_file("query", "backbone", ".fas")
            options().fragment_file = query
            _write_fasta(sequences, query)

    def check_options(self, supply=[]):
        self.check_outputprefix()
        options().info_file = "A_dummy_value"

        ### TIPP
        if options().reference_pkg is not None:
            self.load_reference(
                os.path.join(options().reference.path,
                    '%s.refpkg/' % options().reference_pkg))
        if (options().taxonomy_file is None):
            supply = supply + ["taxonomy file"]
        if options().taxonomy_name_mapping_file is None:
            supply = supply + ["taxonomy  name mapping file"]
        #ExhaustiveAlgorithm.check_options(self, supply)

        ### Back to UPP
        # Check to see if tree/alignment/fragment file provided, if not,
        # generate it from sequence file
        if (
                (not options().tree_file is None) and
                (not options().alignment_file is None) and
                (not options().sequence_file is None)
        ):
            options().fragment_file = options().sequence_file
        elif (
                (options().tree_file is None) and
                (options().alignment_file is None) and
                (not options().sequence_file is None)
        ):
            self.generate_backbone()
        else:
            _LOG.error(
                ("Either specify the backbone alignment and tree and query "
                 "sequences or only the query sequences.  Any other "
                 "combination is invalid"))
            exit(-1)
        sequences = MutableAlignment()
        sequences.read_file_object(open(self.options.alignment_file.name))
        backbone_size = sequences.get_num_taxa()
        if options().backbone_size is None:
            options().backbone_size = backbone_size
        assert options().backbone_size == backbone_size, (
                ("Backbone parameter needs to match actual size of backbone; "
                 "backbone parameter:%s backbone_size:%s")
                % (options().backbone_size, backbone_size))
        if options().placement_size is None:
            options().placement_size = options().backbone_size

        if options().backtranslation_sequence_file and \
                options().molecule != "amino":
            _LOG.error(
                ("Backtranslation can be performed only when "
                 "input sequences are amino acid. "))
            exit(-1)

        return ExhaustiveAlgorithm.check_options(self, supply)

    def merge_results(self):
        assert \
            len(self.root_problem.get_children()) == 1, \
            "Currently UPP works with only one placement subset."
        '''
        Merge alignment subset extended alignments to get one extended
        alignment for current placement subset.
        '''
        pp = self.root_problem.get_children()[0]
        _LOG.info(
            "Merging sub-alignments for placement problem : %s." % pp.label)
        ''' First assign fragments to the placement problem'''
        pp.fragments = pp.parent.fragments.get_soft_sub_alignment([])
        for ap in pp.get_children():
            pp.fragments.seq_names |= set(ap.fragments)

        ''' Then Build an extended alignment by merging all hmmalign results'''
        _LOG.debug(
            "fragments are %d:\n %s" % (
                len(pp.fragments.seq_names), pp.fragments.seq_names))
        extendedAlignment = ExtendedAlignment(pp.fragments.seq_names)
        for ap in pp.children:
            assert isinstance(ap, SeppProblem)
            ''' Get all fragment chunk alignments for this alignment subset'''
            aligned_files = [fp.get_job_result_by_name('hmmalign') for
                             fp in ap.children if
                             fp.get_job_result_by_name('hmmalign') is not None]
            _LOG.debug(
                "Merging fragment chunks for subalignment : %s." % ap.label)
            ap_alg = ap.read_extendend_alignment_and_relabel_columns(
                ap.jobs["hmmbuild"].infile, aligned_files)
            _LOG.debug(
                "Merging alignment subset into placement subset: %s." %
                ap.label)
            extendedAlignment.merge_in(ap_alg, convert_to_string=False)

        extendedAlignment.from_bytearray_to_string()
        self.results = extendedAlignment

        ## TIPP
        mergeinput = []
        '''Append main tree to merge input'''
        mergeinput.append("%s;" % (self.root_problem.subtree.compose_newick(labels=True)))
        for pp in self.root_problem.get_children():
            assert isinstance(pp, SeppProblem)
            for i in range(0, self.root_problem.fragment_chunks):
                if (pp.get_job_result_by_name(
                    get_placement_job_name(i)) is None):
                    continue
                '''append subset trees and json locations to merge input'''
                mergeinput.append(
                    "%s;\n%s" % (
                        pp.subtree.compose_newick(labels=True),
                        pp.get_job_result_by_name(get_placement_job_name(i))))
        mergeinput.append("")
        mergeinput.append("")
        mergeinputstring = "\n".join(mergeinput)
        merge_json_job = self.get_merge_job(mergeinputstring)
        merge_json_job.run()


    def output_results(self):
        extended_alignment = self.results
        _LOG.info("Generating output. ")
        outfilename = self.get_output_filename("alignment.fasta")
        extended_alignment.write_to_path(outfilename)
        _LOG.info("Unmasked alignment written to %s" % outfilename)
        outfilename = self.get_output_filename("insertion_columns.txt")
        extended_alignment.write_insertion_column_indexes(outfilename)
        _LOG.info("The index of insertion columns written to %s" % outfilename)
        if self.options.backtranslation_sequence_file:
            outfilename = self.get_output_filename(
                "backtranslated_alignment.fasta")
            backtranslation_seqs = MutableAlignment()
            backtranslation_seqs.read_file_object(
                self.options.backtranslation_sequence_file)
            try:
                extended_backtranslated_alignment = backtranslate(
                    self.results, backtranslation_seqs)
            except Exception as e:
                _LOG.warning("Backtranslation failed due "
                             "to following error: " + str(e) + ".\n"
                             "No translated DNA sequence will be "
                             "written to a file.")
                pass
            else:
                extended_backtranslated_alignment.write_to_path(outfilename)
                _LOG.info(
                    "Backtranslated alignment written to %s" % outfilename)
                extended_backtranslated_alignment.remove_insertion_columns()
                outfilename = self.get_output_filename(
                    "backtranslated_alignment_masked.fasta")
                extended_backtranslated_alignment.write_to_path(outfilename)
                _LOG.info(
                    "Backtranslated masked alignment written "
                    "to %s" % outfilename)

        extended_alignment.remove_insertion_columns()
        outfilename = self.get_output_filename("alignment_masked.fasta")
        extended_alignment.write_to_path(outfilename)
        _LOG.info("Masked alignment written to %s" % outfilename)
        
        ### from sepp
        namerev_script = self.root_problem.subtree.rename_script()
        if namerev_script:
            outfilename = self.get_output_filename("rename-json.py")
            with open(outfilename, "w") as s:
                s.write(namerev_script)


    def check_and_set_sizes(self, total):
        assert (self.options.placement_size is None) or (
                self.options.placement_size >= total), \
            ("currently UPP works with only one placement subset."
             " Please leave placement subset size option blank.")
        ExhaustiveAlgorithm.check_and_set_sizes(self, total)
        self.options.placement_size = total

    def _get_new_Join_Align_Job(self):
        ### TIPP edit
        return TIPPJoinAlignJobs(self.placer)

    def modify_tree(self, a_tree):
        """ Filter out taxa on long branches """
        self.filtered_taxa = []
        if self.options.long_branch_filter is not None:
            tr = a_tree.get_tree()
            elen = {}
            for e in tr.leaf_edge_iter():
                elen[e] = e.length
            elensort = sorted(elen.values())
            mid = elensort[len(elensort) // 2]
            torem = []
            for k, v in list(elen.items()):
                if v > mid * self.options.long_branch_filter:
                    self.filtered_taxa.append(k.head_node.taxon.label)
                    torem.append(k.head_node.taxon)
            tr.prune_taxa(torem)

    def create_fragment_files(self):
        alg_subset_count = len(list(self.root_problem.iter_leaves()))
        frag_chunk_count = lcm(
            alg_subset_count, self.options.cpu) // alg_subset_count
        _LOG.info(
            "%d taxa pruned from backbone and added to fragments: %s"
            % (len(self.filtered_taxa), " , ".join(self.filtered_taxa)))
        return self.read_and_divide_fragments(
            frag_chunk_count,
            extra_frags=self.root_problem.subalignment.get_soft_sub_alignment(
                self.filtered_taxa))

    ### TIPP difference: skipping def build_jobs(self), since it seems this is
    # mostly to check placer strategy
    def build_jobs(self):
        assert isinstance(self.root_problem, RootProblem)
        for placement_problem in self.root_problem.get_children():
            '''create placer jobs'''
            for i in range(0, self.root_problem.fragment_chunks):
                pj = None
                if self.placer == 'pplacer':
                    pj = PplacerJob()
                    pj.partial_setup_for_subproblem(
                        placement_problem, self.options.info_file, i)
                elif self.placer == 'epa':
                    raise ValueError("EPA currently not supported.")
                placement_problem.add_job(get_placement_job_name(i), pj)
        for alg_problem in placement_problem.children:
            assert isinstance(alg_problem, SeppProblem)
            '''create the build model job'''
            bj = HMMBuildJob()
            bj.setup_for_subproblem(alg_problem, molecule = self.molecule)
            alg_problem.add_job(bj.job_type, bj)
            '''create the search jobs'''
            for fc_problem in alg_problem.get_children():
                sj = HMMSearchJob()
                sj.partial_setup_for_subproblem(
                    fc_problem.fragments, fc_problem, self.elim,
                    self.filters)
                fc_problem.add_job(sj.job_type, sj)
                '''create the align job'''
                aj = HMMAlignJob()
                fc_problem.add_job(aj.job_type, aj)
                aj.partial_setup_for_subproblem(
                    fc_problem, molecule=self.molecule)

    ### TIPP difference: introduce the threshold check
    def connect_jobs(self): 
        ''' a callback function called after hmmbuild jobs are finished '''
        def enq_job_searchfragment(result, search_job):
            search_job.hmmmodel = result
            JobPool().enqueue_job(search_job)
        assert isinstance(self.root_problem, SeppProblem)
        for placement_problem in self.root_problem.get_children():
            ''' for each alignment subproblem, ...'''
            for alg_problem in placement_problem.children:
                assert isinstance(alg_problem, SeppProblem)
                ''' create the build model job '''
                bj = alg_problem.jobs["hmmbuild"]
                ''' create the search jobs '''
                for fc_problem in alg_problem.get_children():
                    sj = fc_problem.jobs["hmmsearch"]
                    '''connect build and search jobs'''
                    bj.add_call_Back(
                        lambda result, next_job=sj: enq_job_searchfragment(
                            result, next_job))
                '''join all align jobs of a placement subset
                    (enqueues placement job)'''
                jaj = self._get_new_Join_Align_Job()
                jaj.setup_with_placement_problem(placement_problem)
            '''Join all search jobs together (enqueues align jobs)'''

            jsj = TIPPJoinSearchJobs(self.alignment_threshold)
            jsj.setup_with_root_problem(self.root_problem)

    ### TIPP difference: this function doesn't exist in UPP, obviously
    def load_reference(self, reference_pkg):
        file = open(reference_pkg + "CONTENTS.json")
        result = json.load(file)
        file.close()
        options().taxonomy_name_mapping_file = open(
            reference_pkg + result['files']['seq_info'])
        options().taxonomy_file = open(
            reference_pkg + result['files']['taxonomy'])
        options().alignment_file = open(
            reference_pkg + result['files']['aln_fasta'])
        options().tree_file = open(reference_pkg + result['files']['tree'])
        options().info_file = reference_pkg + result['files']['tree_stats']

    ### TIPP 
    def read_alignment_and_tree(self):
        (alignment, tree) = AbstractAlgorithm.read_alignment_and_tree(self)
        return alignment, tree

    ### TIPP difference: not in UPP or SEPP; this is a new helper method
    # probably because the setup_for_tipp() is long
    def get_merge_job(self, mergeinputstring):
        merge_json_job = TIPPMergeJsonJob()
        merge_json_job.setup_for_tipp(
            mergeinputstring,
            self.get_output_filename("placement.json"),
            self.options.taxonomy_file,
            self.options.taxonomy_name_mapping_file,
            self.options.placement_threshold,
            self.get_output_filename("classification.txt"),
            self.push_down,
            self.options.distribution,
            self.options.cutoff)
        return merge_json_job

    ### TIPP difference: new in TIPP
    def get_alignment_decomposition_tree(self, p_tree):
        assert isinstance(p_tree, PhylogeneticTree)
        if self.options.alignment_decomposition_tree is None:
            return PhylogeneticTree(Tree(p_tree.den_tree))
        elif p_tree.count_leaves() != self.root_problem.subtree.count_leaves():
            raise ValueError(
                ("Alignment decomposition tree can be different from placement"
                " tree only if placement subset size is set to the number of"
                " taxa (i.e. entire tree)"))
        else:
            _LOG.info("Reading alignment decomposition input tree: %s" % (
                self.options.alignment_decomposition_tree))
            return PhylogeneticTree(
                dendropy.Tree.get_from_stream(
                    self.options.alignment_decomposition_tree,
                    schema="newick",
                    preserve_underscores=True,
                    taxon_set = self.root_problem.subtree.get_tree().taxon_set))

### TIPP difference: exhaustive imports a MergeJsonJob from sepp.jobs that is similar
# UPP presumably inherits this
def TIPPMergeJsonJob(ExternalSeppJob):
    def __init__(self, **kwargs):
        self.job_type = 'jsonmerger'
        ExternalSeppJob.__init__(self, self.job_type, **kwargs)
        self.out_file = None
        self.input_string = None # from MergeJsonJob

        self.distribution = False
        self.taxonomy = None
        self.mapping = None
        self.threshold = None
        self.classification_file = None
        self.elim = float(options().hmmsearch.elim)
        
        if options().hmmsearch.filters.upper() == "TRUE":
            self.filters = True
        else:
            if options().hmmsearch.filters.upper() == "FALSE":
                self.filters = False
            else:
                self.filters = None
        if self.filters is None:
            raise Exception(
                "Expecting true/false for options().hmmsearch/filters")
        self.strategy = options().exhaustive.strategy 
        self.minsubsetsize = int(options().exhaustive.minsubsetsize)
        self.alignment_threshold = float(options().alignment_threshold)
        self.molecule = options().molecule
        self.placer = options().exhaustive.__dict__['placer'].lower()
        self.cutoff = 0
        self.push_down = False
    
    def setup(self, in_string, output_file, **kwargs):
        self.stdindata = in_string
        self.out_file = output_file
        self._kwargs = kwargs

    ### TIPP difference: new function
    def setup_for_tipp(self, in_string, output_file, taxonomy, mapping, 
                      threshold, classification_file, push_down,
                      distribution=False, cutoff=0, **kwargs):
        self.stdindata = in_string
        self.out_file = output_file

        # tipp files
        self.taxonomy = taxonomy.name
        self.mapping = mapping.name
        self.distribution = distribution
        self.threshold = str(threshold)
        self.classification_file = classification_file
        self.push_down = push_down
        self._kwargs = kwargs
        self.cutoff = cutoff

    def get_invocation(self):

        # the -r 4 is new
        invoc = ["java", "-jar", self.path, "-", "-", self.out_file, "-r", "4"]

        # all new invocations
        if self.taxonomy is not None:
            invoc.extend(['-t', self.taxonomy])
        if self.mapping is not None:
            invoc.extend(['-m', self.mapping])
        if self.threshold is not None:
            invoc.extend(['-p', self.threshold])
        if self.classification_file is not None:
            invoc.extend(['-c', self.classification_file])
        if self.distribution:
            invoc.extend(['-d'])
        if not self.push_down:
            invoc.extend(['-u'])
        invoc.extend(['-C', str(self.cutoff)])

        return invoc
    
    def characterize_input(self):
        return "input:pipe output:%s; Pipe:\n%s" % (
            self.out_file, self.stdindata)

    # this is the same
    def read_results(self):
        """
        Since the output file can be huge, we don't want to read it here,
        because it will need to get pickled and unpickled. Instead, we just
        send back the file name, and will let the caller figure out what to do
        with it.
        """
        assert os.path.exists(self.out_file)
        assert os.stat(self.out_file)[stat.ST_SIZE] != 0
        return self.out_file, self.stdoutdata


def augment_parser():
    root_p = open(os.path.join(os.path.split(
        os.path.split(__file__)[0])[0], "home.path")).readlines()[0].strip()
    ### TIPP difference: adding the tipp config path
    tipp_config_path = os.path.join(root_p, "tipp.config")
    upp_config_path = os.path.join(root_p, "upp.config")
    sepp.config.set_main_config_path(tipp_config_path)
    parser = sepp.config.get_parser()
    parser.description = (
        "This script runs the TIPP3 algorithm on set of sequences.") # TODO: Add inputs to comment. 
        #A backbone"
        #" alignment and tree can be given as input.  If none is provided, a"
        #" backbone will be automatically generated.")

    decompGroup = parser.groups['decompGroup']
    decompGroup.__dict__['description'] = ' '.join(
        ["These options",
         ("determine the alignment decomposition size, backbone size,"
          " and how to decompose the backbone set.")])

    decompGroup.add_argument(
        "-A", "--alignmentSize", type=int,
        dest="alignment_size", metavar="N",
        default=10,
        help="max alignment subset size of N "
             "[default: 10]")
    decompGroup.add_argument(
        "-R", "--full_length_range", type=str,
        dest="full_length_range", metavar="\"Nmin Nmax\"",
        default=None,
        help="Only consider sequences with lengths within Nmin and Nmax")
    decompGroup.add_argument(
        "-M", "--median_full_length", type=int,
        dest="median_full_length", metavar="N",
        default=None,
        help="Consider all fragments that are 25%% longer or shorter than N "
             "to be excluded from the backbone.  If value is -1, then UPP will"
             " use the median of the sequences as the median full length "
             "[default: None]")
    decompGroup.add_argument(
        "-T", "--backbone_threshold", type=float,
        dest="backbone_threshold", metavar="N",
        default=0.25,
        help="Only consider sequences with lengths within (1-N)*M and (1+N)*"
             "M as full-length, where M is the median length of the full-"
             "length sequence given by the -M option and N is the percentage"
             " given by the -T option."
             "[default: 0.25]")
    decompGroup.add_argument(
        "-B", "--backboneSize", type=int,
        dest="backbone_size", metavar="N",
        default=None,
        help="(Optional) size of backbone set.  "
             "If no backbone tree and alignment is given, the sequence file "
             "will be randomly split into a backbone set (size set to N) and "
             "query set (remaining sequences), [default: min(1000,input "
             "size)]")
    decompGroup.add_argument(
        "-S", "--decomp_strategy", type=valid_decomp_strategy,
        dest="decomp_strategy", metavar="DECOMP",
        default="hierarchical",
        help="decomposition strategy "
             "[default: ensemble of HMMs (hierarchical)]")

    inputGroup = parser.groups['inputGroup']
    inputGroup.add_argument(
        "-s", "--sequence_file", type=argparse.FileType('r'),
        dest="sequence_file", metavar="SEQ",
        default=None,
        help="Unaligned sequence file.  "
             "If no backbone tree and alignment is given, the sequence file "
             "will be randomly split into a backbone set (size set to B) and "
             "query set (remaining sequences), [default: None]")
    inputGroup.add_argument(
        "-c", "--config",
        dest="config_file", metavar="CONFIG",
        type=argparse.FileType('r'),
        help="A config file, including options used to run UPP. Options "
             "provided as command line arguments overwrite config file values"
             " for those options. "
             "[default: %(default)s]")
    inputGroup.add_argument(
        "-t", "--tree",
        dest="tree_file", metavar="TREE",
        type=argparse.FileType('r'),
        help="Input tree file (newick format) "
             "[default: %(default)s]")
    inputGroup.add_argument(
        "-a", "--alignment",
        dest="alignment_file", metavar="ALIGN",
        type=argparse.FileType('r'),
        help="Aligned fasta file "
             "[default: %(default)s]")

    inputGroup.add_argument(
        "-b", "--backtranslation",
        dest="backtranslation_sequence_file", metavar="SEQ",
        type=argparse.FileType('r'),
        default=None,
        help="Fasta file containing unaligned DNA sequences "
             "corresponding every reference and query sequence "
             "[default: None]")

    uppGroup = parser.add_argument_group(
        "UPP Options".upper(),
        "These options set settings specific to UPP")

    uppGroup.add_argument(
        "-l", "--longbranchfilter", type=int,
        dest="long_branch_filter", metavar="N",
        default=None,
        help="Branches longer than N times the median branch length are "
             "filtered from backbone and added to fragments."
             " [default: None (no filtering)]")

    seppGroup = parser.add_argument_group(
        "SEPP Options".upper(),
        ("These options set settings specific to SEPP and are not used"
         " for UPP."))
    seppGroup.add_argument(
        "-P", "--placementSize", type=int,
        dest="placement_size", metavar="N",
        default=None,
        help="max placement subset size of N "
             "[default: 10%% of the total number of taxa]")
    seppGroup.add_argument(
        "-r", "--raxml",
        dest="info_file", metavar="RAXML",
        type=argparse.FileType('r'),
        help="RAxML_info file including model parameters, generated by RAxML."
             "[default: %(default)s]")
    seppGroup.add_argument(
        "-f", "--fragment",
        dest="fragment_file", metavar="FRAG",
        type=argparse.FileType('r'),
        help="fragment file "
             "[default: %(default)s]")

    tippGroup = parser.add_argument_group(
        "TIPP Options".upper(),
        "These arguments set settings specific to TIPP")
    tippGroup.add_argument(
        "-R", "--reference_pkg", type=str,
        dest="reference_pkg", metavar="N",
        default=None,
        help="Use a pre-computed reference package [default: None]")
    tippGroup.add_argument(
        "-at", "--alignmentThreshold", type=float,
        dest="alignment_threshold", metavar="N",
        default=0.95,
        help="Enough alignment subsets are selected to reach a "
        "cumulative probablity of N. This should be a number"
             "between 0 and 1 [default 0.95]")
    tippGroup.add_argument(
        "-D", "--dist",
        dest="distribution", action="store_true",
        default=False,
        help="Treat fragments as distribution"
    )
    tippGroup.add_argument(
        "-pt", "--placementThreshold", type=float,
        dest="placement_threshold", metavar="N",
        default=0.95,
        help="Enough placements are selected to reach a "
        "cumulative probability of N. This should be a number"
        " between 0 and 1 [default: 0.95]")
    tippGroup.add_argument(
        "-PD", "--push_down", type=bool,
        dest="push_down", metavar="N",
        default=True,
        help="Whether to classify based on children below or above"
             " insertion point. [default: True]")
    tippGroup.add_argument(
        "-tx", "--taxonomy", type=argparse.FileType('r'),
        dest="taxonomy_file", metavar="TAXONOMY",
        help="A file describing the taxonomy. This is a comma-separated text "
             "file that has the following fileds: "
             " taxon_id, parent_id, taxon_name, rank."
             " If there are other columns, they are ingored."
             " The first line is also assumed as the header, and ignored.")
    tippGroup.add_argument(
        "-txm", "--taxonomyNameMapping", type=argparse.FileType('r'),
        dest="taxonomy_name_mapping_file", metavar="MAPPING",
        help="A comma-separated text file mapping alignment sequence names to "
             " taxonomic ids. "
             " Formats (each line): "
             " sequence_name, taxon_id. "
             " If there are other columns, they are ignored. The first line is "
             " also assumed to be the header, and ignored.")
    tippGroup.add_argument(
        "-adt", "--alignmentDecompositionTree", type=argparse.FileType('r'),
        dest="alignment_decomposition_tree", metavar="TREE", default=None,
        help="A newick tree file used for decomposing taxa into alignment subsets."
             "[default: the backbone tree]")
    tippGroup.add_argument(
        "-C", "--cutoff", type=float,
        dest="cutoff", metavar="N",
        default=0.0,
        help="Placement probability requirement to count toward the distribution."
             "This should be a number between 0 and 1 [default: 0.0]") 

def main():
    augment_parser()
    TIPPExhaustiveAlgorithm().run()


if __name__ == '__main__':
    main()
