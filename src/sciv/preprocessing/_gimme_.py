# -*- coding: UTF-8 -*-

import os
from typing import Union, Tuple, Literal

import tempfile

from pandas import DataFrame
import pandas as pd


from .. import util as ul

__name__: str = "preprocessing_gimme"

_Genome = Literal['hg38', 'hg19']


class RunGimme:

    def __init__(
        self,
        genomes_path: str,
        tf_name_list: list = None,
        fpr: float = 0.01,
        columns: Union[list, tuple] = ("chr", "start", "end"),
        peak_split_character: tuple = (":", "-"),
        genomes: tuple[_Genome, _Genome] = ("hg38", "hg19")
    ):
        from gimmemotifs.motif import read_motifs

        self.genomes_path = genomes_path
        self.columns = columns
        self.peak_split_character = peak_split_character
        self.tf_name_list = tf_name_list
        self.fpr = fpr
        self.log = ul.log(__name__)

        # Load motif database (default to use motifs provided by GimmeMotifs)
        self.log.info("Load motif database")
        self.motifs = read_motifs()

        if tf_name_list is not None:
            self.motifs = [m for m in self.motifs if any(f in m.factors['direct'] for f in tf_name_list)]

        self.hg19_scanner = self.get_hg19_scanner()
        self.hg38_scanner = self.get_hg38_scanner()

        self.hg19_fasta = self.get_genome_fa("hg19")
        self.hg38_fasta = self.get_genome_fa("hg38")

    def create_tmp_fa_file(self, genome: str, peak_df: DataFrame) -> str:
        from Bio import SeqIO
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord

        self.log.info("Create tmp fasta file")

        if genome == "hg19":
            get_genome_seq = self.get_hg19_seq
        elif genome == "hg38":
            get_genome_seq = self.get_hg38_seq
        else:
            raise ValueError(f"Unknown genome: {genome}")

        records = []

        for _chr_, _start_, _end_ in zip(peak_df[self.columns[0]], peak_df[self.columns[1]], peak_df[self.columns[2]]):
            seq_content = get_genome_seq(chrom=_chr_, start=_start_, end=_end_)
            # Create SeqRecord object
            record = SeqRecord(
                Seq(seq_content),
                id=f"{_chr_}{self.peak_split_character[0]}{_start_}{self.peak_split_character[1]}{_end_}",
                description=""
            )
            records.append(record)

        # Write file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fa') as tmp:
            SeqIO.write(records, tmp, "fasta")
            tmp_path = tmp.name

        return tmp_path

    def get_genome_fa(self, genome: str):
        from gimmemotifs.fasta import Fasta
        self.log.info(f"Load fasta file ==> {genome}")
        return Fasta(os.path.join(self.genomes_path, genome, f"{genome}.fa"))

    def get_hg19_seq(self, chrom: str, start: int, end: int) -> str:
        return self.hg19_fasta[chrom][start:end]

    def get_hg38_seq(self, chrom: str, start: int, end: int) -> str:
        return self.hg38_fasta[chrom][start:end]

    def get_scanner(self, genome: str) -> str:
        from gimmemotifs.scanner import Scanner
        self.log.info(f"Scanner ==> {genome}")
        _scanner_ = Scanner()
        _scanner_.set_motifs(self.motifs)
        _scanner_.set_genome(genome, self.genomes_path)
        _scanner_.set_threshold(fpr=self.fpr)
        return _scanner_

    def get_hg19_scanner(self) -> str:
        return self.get_scanner("hg19")

    def get_hg38_scanner(self) -> str:
        return self.get_scanner("hg38")

    def get_single_seq(self, seq_name: str) -> Tuple:

        if self.peak_split_character[0] == self.peak_split_character[1]:
            _split_ = seq_name.split(self.peak_split_character[0])
            seq_chr = _split_[0]
            seq_start = _split_[1]
            seq_end = _split_[2]
        else:
            _split_ = seq_name.split(self.peak_split_character[0])
            seq_chr = _split_[0]
            _split_ = _split_[1].split(self.peak_split_character[1])
            seq_start = _split_[0]
            seq_end = _split_[1]

        return seq_chr, seq_start, seq_end

    def get_motif_result(self, genome: str, peak_df: DataFrame) -> DataFrame:
        from gimmemotifs.fasta import Fasta

        tmp_fa_file = self.create_tmp_fa_file(genome, peak_df)

        if genome == "hg19":
            genome_scanner = self.hg19_scanner
        elif genome == "hg38":
            genome_scanner = self.hg38_scanner
        else:
            raise ValueError(f"Unknown genome: {genome}")

        seqs = Fasta(tmp_fa_file)
        motif_matches_data = genome_scanner.scan(seqs)

        self.log.info("Motif matches data")

        data_list = []

        for i, result in enumerate(motif_matches_data):
            seq_name: str = seqs.ids[i]
            seq_chr, seq_start, seq_end = self.get_single_seq(seq_name)

            for m, matches in enumerate(result):
                motif = self.motifs[m]
                tf_name_list = motif.factors['direct']

                for score, pos, strand in matches:
                    _strand_ = '+' if strand == 1 else '-'

                    for tf in tf_name_list:

                        if self.tf_name_list is not None:
                            if tf in self.tf_name_list:
                                data_list.append([seq_chr, seq_start, seq_end, str(motif), tf, score, pos, _strand_, seq_name])
                        else:
                            data_list.append([seq_chr, seq_start, seq_end, str(motif), tf, score, pos, _strand_, seq_name])

        motif_matches_df = pd.DataFrame(
            data_list,
            columns=['chr', 'start', 'end', 'motif', 'tf', 'score', 'position', 'strand', 'seq_name']
        )

        if os.path.exists(tmp_fa_file):
            os.remove(tmp_fa_file)

        return motif_matches_df



