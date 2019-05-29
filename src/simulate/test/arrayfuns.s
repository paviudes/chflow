# mark_description "Intel(R) C Intel(R) 64 Compiler for applications running on Intel(R) 64, Version 19.0.1.144 Build 20181018";
# mark_description "--vec-report -S";
	.file "arrayfuns.c"
	.section	__TEXT, __text
L_TXTST0:
L_2__routine_start__Free4D_0:
# -- Begin  _Free4D
	.section	__TEXT, __text
# mark_begin;
       .align    4
	.globl _Free4D
# --- Free4D(complex128_t ****, const int *)
_Free4D:
# parameter 1: %rdi
# parameter 2: %rsi
L_B1.1:                         # Preds L_B1.0
                                # Execution count [1.00e+00]
L_LCFI1:
L____tag_value__Free4D.1:
L_L2:
                                                          #38.54
        subq      $88, %rsp                                     #38.54
L_LCFI2:
        xorl      %ecx, %ecx                                    #41.7
        movslq    (%rsi), %rdx                                  #41.18
        testq     %rdx, %rdx                                    #41.18
        jle       L_B1.16       # Prob 10%                      #41.18
                                # LOE rdx rcx rbx rbp rsi rdi r12 r13 r14 r15
L_B1.2:                         # Preds L_B1.1
                                # Execution count [9.00e-01]
        movslq    4(%rsi), %rax                                 #42.19
        movq      %r12, (%rsp)                                  #42.19[spill]
L_LCFI3:
        movq      %rdi, %r12                                    #42.19
        movq      %r13, 8(%rsp)                                 #42.19[spill]
L_LCFI4:
        movq      %rcx, %r13                                    #42.19
        movq      %r14, 16(%rsp)                                #42.19[spill]
L_LCFI5:
        movq      %rax, %r14                                    #42.19
        movq      %r15, 24(%rsp)                                #42.19[spill]
        movq      %rbx, 32(%rsp)                                #42.19[spill]
L_LCFI6:
        movq      %rdx, %rbx                                    #42.19
        movq      %rbp, 40(%rsp)                                #42.19[spill]
L_LCFI7:
        movq      %rsi, %rbp                                    #42.19
                                # LOE rbx rbp r12 r13 r14
L_B1.3:                         # Preds L_B1.14 L_B1.2
                                # Execution count [5.00e+00]
        xorl      %edx, %edx                                    #42.8
        testq     %r14, %r14                                    #42.19
        jle       L_B1.13       # Prob 10%                      #42.19
                                # LOE rdx rbx rbp r12 r13 r14
L_B1.4:                         # Preds L_B1.3
                                # Execution count [4.50e+00]
        movslq    8(%rbp), %rax                                 #43.20
        movq      (%r12), %r15                                  #44.10
        movq      %rbx, 72(%rsp)                                #44.10[spill]
        movq      %rax, %rbx                                    #44.10
        movq      %r13, 64(%rsp)                                #44.10[spill]
        movq      %r12, 56(%rsp)                                #44.10[spill]
        movq      %rbp, 48(%rsp)                                #44.10[spill]
        movq      %rdx, %rbp                                    #44.10
                                # LOE rbx rbp r14 r15
L_B1.5:                         # Preds L_B1.11 L_B1.4
                                # Execution count [2.50e+01]
        xorl      %r13d, %r13d                                  #43.9
        testq     %rbx, %rbx                                    #43.20
        jle       L_B1.10       # Prob 10%                      #43.20
                                # LOE rbx rbp r13 r14 r15
L_B1.6:                         # Preds L_B1.5
                                # Execution count [2.25e+01]
        movq      (%r15), %r12                                  #44.10
                                # LOE rbx rbp r12 r13 r14 r15
L_B1.7:                         # Preds L_B1.8 L_B1.6
                                # Execution count [1.25e+02]
        movq      (%r12,%r13,8), %rdi                           #44.5
#       free(void *)
        call      _free                                         #44.5
                                # LOE rbx rbp r12 r13 r14 r15
L_B1.8:                         # Preds L_B1.7
                                # Execution count [1.25e+02]
        incq      %r13                                          #43.30
        cmpq      %rbx, %r13                                    #43.20
        jl        L_B1.7        # Prob 82%                      #43.20
                                # LOE rbx rbp r12 r13 r14 r15
L_B1.10:                        # Preds L_B1.8 L_B1.5
                                # Execution count [2.50e+01]
        movq      (%r15), %rdi                                  #45.4
#       free(void *)
        call      _free                                         #45.4
                                # LOE rbx rbp r14 r15
L_B1.11:                        # Preds L_B1.10
                                # Execution count [2.50e+01]
        incq      %rbp                                          #42.29
        addq      $8, %r15                                      #42.29
        cmpq      %r14, %rbp                                    #42.19
        jl        L_B1.5        # Prob 82%                      #42.19
                                # LOE rbx rbp r14 r15
L_B1.12:                        # Preds L_B1.11
                                # Execution count [4.50e+00]
        movq      72(%rsp), %rbx                                #[spill]
        movq      64(%rsp), %r13                                #[spill]
        movq      56(%rsp), %r12                                #[spill]
        movq      48(%rsp), %rbp                                #[spill]
                                # LOE rbx rbp r12 r13 r14
L_B1.13:                        # Preds L_B1.12 L_B1.3
                                # Execution count [5.00e+00]
        movq      (%r12), %rdi                                  #47.3
#       free(void *)
        call      _free                                         #47.3
                                # LOE rbx rbp r12 r13 r14
L_B1.14:                        # Preds L_B1.13
                                # Execution count [5.00e+00]
        incq      %r13                                          #41.28
        addq      $8, %r12                                      #41.28
        cmpq      %rbx, %r13                                    #41.18
        jl        L_B1.3        # Prob 82%                      #41.18
                                # LOE rbx rbp r12 r13 r14
L_B1.15:                        # Preds L_B1.14
                                # Execution count [9.00e-01]
        movq      (%rsp), %r12                                  #[spill]
L_LCFI8:
        movq      8(%rsp), %r13                                 #[spill]
L_LCFI9:
        movq      16(%rsp), %r14                                #[spill]
L_LCFI10:
        movq      24(%rsp), %r15                                #[spill]
L_LCFI11:
        movq      32(%rsp), %rbx                                #[spill]
L_LCFI12:
        movq      40(%rsp), %rbp                                #[spill]
L_LCFI13:
                                # LOE rbx rbp r12 r13 r14 r15
L_B1.16:                        # Preds L_B1.15 L_B1.1
                                # Execution count [1.00e+00]
        addq      $88, %rsp                                     #49.1
L_LCFI14:
        ret                                                     #49.1
        .align    4
                                # LOE
L_LCFI15:
# mark_end;
L..LN_Free4D.0:
	.section	__DATA, __data
# -- End  _Free4D
	.section	__TEXT, __text
L_2__routine_start__Free2D_1:
# -- Begin  _Free2D
	.section	__TEXT, __text
# mark_begin;
       .align    4
	.globl _Free2D
# --- Free2D(complex128_t **, const int *)
_Free2D:
# parameter 1: %rdi
# parameter 2: %rsi
L_B2.1:                         # Preds L_B2.0
                                # Execution count [1.00e+00]
L_LCFI16:
L____tag_value__Free2D.37:
L_L38:
                                                         #51.52
        subq      $24, %rsp                                     #51.52
L_LCFI17:
        xorl      %edx, %edx                                    #54.7
        movslq    (%rsi), %rax                                  #54.18
        testq     %rax, %rax                                    #54.18
        jle       L_B2.6        # Prob 10%                      #54.18
                                # LOE rax rdx rbx rbp rdi r12 r13 r14 r15
L_B2.2:                         # Preds L_B2.1
                                # Execution count [9.00e-01]
        movq      %r12, 16(%rsp)                                #[spill]
L_LCFI18:
        movq      %rax, %r12                                    #
        movq      %r13, 8(%rsp)                                 #[spill]
L_LCFI19:
        movq      %rdx, %r13                                    #
        movq      %r14, (%rsp)                                  #[spill]
L_LCFI20:
        movq      %rdi, %r14                                    #
                                # LOE rbx rbp r12 r13 r14 r15
L_B2.3:                         # Preds L_B2.4 L_B2.2
                                # Execution count [5.00e+00]
        movq      (%r14,%r13,8), %rdi                           #55.3
#       free(void *)
        call      _free                                         #55.3
                                # LOE rbx rbp r12 r13 r14 r15
L_B2.4:                         # Preds L_B2.3
                                # Execution count [5.00e+00]
        incq      %r13                                          #54.28
        cmpq      %r12, %r13                                    #54.18
        jl        L_B2.3        # Prob 82%                      #54.18
                                # LOE rbx rbp r12 r13 r14 r15
L_B2.5:                         # Preds L_B2.4
                                # Execution count [9.00e-01]
        movq      16(%rsp), %r12                                #[spill]
L_LCFI21:
        movq      8(%rsp), %r13                                 #[spill]
L_LCFI22:
        movq      (%rsp), %r14                                  #[spill]
L_LCFI23:
                                # LOE rbx rbp r12 r13 r14 r15
L_B2.6:                         # Preds L_B2.5 L_B2.1
                                # Execution count [1.00e+00]
        addq      $24, %rsp                                     #56.1
L_LCFI24:
        ret                                                     #56.1
        .align    4
                                # LOE
L_LCFI25:
# mark_end;
L..LN_Free2D.1:
	.section	__DATA, __data
# -- End  _Free2D
	.section	__TEXT, __text
L_2__routine_start__Print4D_2:
# -- Begin  _Print4D
	.section	__TEXT, __text
# mark_begin;
       .align    4
	.globl _Print4D
# --- Print4D(complex128_t ****, const int *)
_Print4D:
# parameter 1: %rdi
# parameter 2: %rsi
L_B3.1:                         # Preds L_B3.0
                                # Execution count [1.00e+00]
L_LCFI26:
L____tag_value__Print4D.53:
L_L54:
                                                         #58.55
        subq      $88, %rsp                                     #58.55
L_LCFI27:
        xorl      %edx, %edx                                    #61.7
        movslq    (%rsi), %rcx                                  #61.18
        testq     %rcx, %rcx                                    #61.18
        jle       L_B3.18       # Prob 10%                      #61.18
                                # LOE rdx rcx rbx rbp rsi rdi r12 r13 r14 r15
L_B3.2:                         # Preds L_B3.1
                                # Execution count [9.00e-01]
        movslq    4(%rsi), %rax                                 #62.19
        movq      %r12, 40(%rsp)                                #62.19[spill]
L_LCFI28:
        movq      %rsi, %r12                                    #62.19
        movq      %r13, 32(%rsp)                                #62.19[spill]
L_LCFI29:
        movq      %rdi, %r13                                    #62.19
        movq      %r14, 24(%rsp)                                #62.19[spill]
        movq      %r15, 16(%rsp)                                #62.19[spill]
        movq      %rbx, 8(%rsp)                                 #62.19[spill]
L_LCFI30:
        movq      %rdx, %rbx                                    #62.19
        movq      %rbp, (%rsp)                                  #62.19[spill]
L_LCFI31:
                                # LOE rax rcx rbx r12 r13
L_B3.3:                         # Preds L_B3.16 L_B3.2
                                # Execution count [5.00e+00]
        xorl      %r8d, %r8d                                    #62.8
        testq     %rax, %rax                                    #62.19
        jle       L_B3.16       # Prob 10%                      #62.19
                                # LOE rax rcx rbx r8 r12 r13
L_B3.4:                         # Preds L_B3.3
                                # Execution count [4.50e+00]
        movl      %ebx, %ebp                                    #65.6
        movslq    8(%r12), %rcx                                 #63.20
        movl      %ebp, 56(%rsp)                                #65.6[spill]
                                # LOE rax rcx rbx r8 r12 r13
L_B3.5:                         # Preds L_B3.14 L_B3.4
                                # Execution count [2.50e+01]
        xorl      %ebp, %ebp                                    #63.9
        testq     %rcx, %rcx                                    #63.20
        jle       L_B3.14       # Prob 10%                      #63.20
                                # LOE rax rcx rbx rbp r8 r12 r13
L_B3.6:                         # Preds L_B3.5
                                # Execution count [2.25e+01]
        movl      %r8d, %r9d                                    #65.6
        movslq    12(%r12), %rax                                #64.21
        movl      %r9d, 48(%rsp)                                #65.6[spill]
        movq      %r8, 72(%rsp)                                 #65.6[spill]
                                # LOE rax rcx rbx rbp r12 r13
L_B3.7:                         # Preds L_B3.12 L_B3.6
                                # Execution count [1.25e+02]
        xorl      %r14d, %r14d                                  #64.10
        xorl      %r15d, %r15d                                  #64.21
        testq     %rax, %rax                                    #64.21
        jle       L_B3.12       # Prob 10%                      #64.21
                                # LOE rax rcx rbx rbp r12 r13 r14 r15
L_B3.8:                         # Preds L_B3.7
                                # Execution count [1.12e+02]
        movl      %ebp, %eax                                    #65.6
        movl      %eax, 64(%rsp)                                #65.6[spill]
                                # LOE rbx rbp r12 r13 r14 r15
L_B3.9:                         # Preds L_B3.10 L_B3.8
                                # Execution count [6.25e+02]
        movq      (%r13,%rbx,8), %r9                            #65.68
        lea       L_2__STRING.0(%rip), %rdi                     #65.6
        movq      72(%rsp), %r10                                #65.68[spill]
        movl      %r14d, %r8d                                   #65.6
        movl      $2, %eax                                      #65.6
        movl      56(%rsp), %esi                                #65.6[spill]
        movq      (%r9,%r10,8), %r11                            #65.68
        movl      48(%rsp), %edx                                #65.6[spill]
        movq      (%r11,%rbp,8), %rcx                           #65.68
        movsd     (%r15,%rcx), %xmm0                            #65.68
        movsd     8(%r15,%rcx), %xmm1                           #65.68
        movl      64(%rsp), %ecx                                #65.6[spill]
L____tag_value__Print4D.75:
#       printf(const char *, ...)
        call      _printf                                       #65.6
L____tag_value__Print4D.76:
                                # LOE rbx rbp r12 r13 r14 r15
L_B3.10:                        # Preds L_B3.9
                                # Execution count [6.25e+02]
        incq      %r14                                          #64.31
        addq      $16, %r15                                     #64.31
        movslq    12(%r12), %rax                                #64.21
        cmpq      %rax, %r14                                    #64.21
        jl        L_B3.9        # Prob 82%                      #64.21
                                # LOE rax rbx rbp r12 r13 r14 r15
L_B3.11:                        # Preds L_B3.10
                                # Execution count [1.13e+02]
        movslq    8(%r12), %rcx                                 #63.20
                                # LOE rax rcx rbx rbp r12 r13
L_B3.12:                        # Preds L_B3.11 L_B3.7
                                # Execution count [1.25e+02]
        incq      %rbp                                          #63.30
        cmpq      %rcx, %rbp                                    #63.20
        jl        L_B3.7        # Prob 82%                      #63.20
                                # LOE rax rcx rbx rbp r12 r13
L_B3.13:                        # Preds L_B3.12
                                # Execution count [2.25e+01]
        movq      72(%rsp), %r8                                 #[spill]
        movslq    4(%r12), %rax                                 #62.19
                                # LOE rax rcx rbx r8 r12 r13
L_B3.14:                        # Preds L_B3.13 L_B3.5
                                # Execution count [2.50e+01]
        incq      %r8                                           #62.29
        cmpq      %rax, %r8                                     #62.19
        jl        L_B3.5        # Prob 82%                      #62.19
                                # LOE rax rcx rbx r8 r12 r13
L_B3.15:                        # Preds L_B3.14
                                # Execution count [4.50e+00]
        .byte     15                                            #61.18
        .byte     31                                            #61.18
        .byte     64                                            #61.18
        .byte     0                                             #61.18
        movslq    (%r12), %rcx                                  #61.18
                                # LOE rax rcx rbx r12 r13
L_B3.16:                        # Preds L_B3.15 L_B3.3
                                # Execution count [5.00e+00]
        incq      %rbx                                          #61.28
        cmpq      %rcx, %rbx                                    #61.18
        jl        L_B3.3        # Prob 82%                      #61.18
                                # LOE rax rcx rbx r12 r13
L_B3.17:                        # Preds L_B3.16
                                # Execution count [9.00e-01]
        movq      40(%rsp), %r12                                #[spill]
L_LCFI32:
        movq      32(%rsp), %r13                                #[spill]
L_LCFI33:
        movq      24(%rsp), %r14                                #[spill]
L_LCFI34:
        movq      16(%rsp), %r15                                #[spill]
L_LCFI35:
        movq      8(%rsp), %rbx                                 #[spill]
L_LCFI36:
        movq      (%rsp), %rbp                                  #[spill]
L_LCFI37:
                                # LOE rbx rbp r12 r13 r14 r15
L_B3.18:                        # Preds L_B3.17 L_B3.1
                                # Execution count [1.00e+00]
        lea       il0_peep_printf_format_0(%rip), %rdi          #66.2
        addq      $88, %rsp                                     #66.2
L_LCFI38:
        jmp       _puts                                         #66.2
        .align    4
                                # LOE
L_LCFI39:
# mark_end;
L..LN_Print4D.2:
	.cstring
	.align 2
	.align 2
il0_peep_printf_format_0:
	.long	2021161080
	.word	30840
	.byte	0
	.section	__DATA, __data
# -- End  _Print4D
	.section	__TEXT, __text
L_2__routine_start__Print2D_3:
# -- Begin  _Print2D
	.section	__TEXT, __text
# mark_begin;
       .align    4
	.globl _Print2D
# --- Print2D(complex128_t **, const int *)
_Print2D:
# parameter 1: %rdi
# parameter 2: %rsi
L_B4.1:                         # Preds L_B4.0
                                # Execution count [1.00e+00]
L_LCFI40:
L____tag_value__Print2D.92:
L_L93:
                                                         #69.53
        subq      $56, %rsp                                     #69.53
L_LCFI41:
        xorl      %ecx, %ecx                                    #72.7
        movslq    (%rsi), %rax                                  #72.18
        testq     %rax, %rax                                    #72.18
        jle       L_B4.10       # Prob 10%                      #72.18
                                # LOE rax rcx rbx rbp rsi rdi r12 r13 r14 r15
L_B4.2:                         # Preds L_B4.1
                                # Execution count [9.00e-01]
        movslq    4(%rsi), %rdx                                 #73.19
        movq      %r12, 40(%rsp)                                #73.19[spill]
        movq      %r13, 32(%rsp)                                #73.19[spill]
L_LCFI42:
        movq      %rdi, %r13                                    #73.19
        movq      %r14, 24(%rsp)                                #73.19[spill]
L_LCFI43:
        movq      %rsi, %r14                                    #73.19
        movq      %r15, 16(%rsp)                                #73.19[spill]
        movq      %rbx, 8(%rsp)                                 #73.19[spill]
        movq      %rbp, (%rsp)                                  #73.19[spill]
L_LCFI44:
        movq      %rcx, %rbp                                    #73.19
                                # LOE rax rdx rbp r13 r14
L_B4.3:                         # Preds L_B4.8 L_B4.2
                                # Execution count [5.00e+00]
        xorl      %r12d, %r12d                                  #73.8
        xorl      %r15d, %r15d                                  #73.19
        testq     %rdx, %rdx                                    #73.19
        jle       L_B4.8        # Prob 10%                      #73.19
                                # LOE rax rdx rbp r12 r13 r14 r15
L_B4.4:                         # Preds L_B4.3
                                # Execution count [4.50e+00]
        movl      %ebp, %ebx                                    #74.4
                                # LOE rbp r12 r13 r14 r15 ebx
L_B4.5:                         # Preds L_B4.6 L_B4.4
                                # Execution count [2.50e+01]
        movq      (%r13,%rbp,8), %r8                            #74.50
        lea       L_2__STRING.2(%rip), %rdi                     #74.4
        movl      %ebx, %esi                                    #74.4
        movl      %r12d, %edx                                   #74.4
        movl      $2, %eax                                      #74.4
        movsd     (%r15,%r8), %xmm0                             #74.50
        movsd     8(%r15,%r8), %xmm1                            #74.50
L____tag_value__Print2D.107:
#       printf(const char *, ...)
        call      _printf                                       #74.4
L____tag_value__Print2D.108:
                                # LOE rbp r12 r13 r14 r15 ebx
L_B4.6:                         # Preds L_B4.5
                                # Execution count [2.50e+01]
        incq      %r12                                          #73.29
        addq      $16, %r15                                     #73.29
        movslq    4(%r14), %rdx                                 #73.19
        cmpq      %rdx, %r12                                    #73.19
        jl        L_B4.5        # Prob 82%                      #73.19
                                # LOE rdx rbp r12 r13 r14 r15 ebx
L_B4.7:                         # Preds L_B4.6
                                # Execution count [4.50e+00]
        movslq    (%r14), %rax                                  #72.18
                                # LOE rax rdx rbp r13 r14
L_B4.8:                         # Preds L_B4.7 L_B4.3
                                # Execution count [5.00e+00]
        incq      %rbp                                          #72.28
        cmpq      %rax, %rbp                                    #72.18
        jl        L_B4.3        # Prob 82%                      #72.18
                                # LOE rax rdx rbp r13 r14
L_B4.9:                         # Preds L_B4.8
                                # Execution count [9.00e-01]
        movq      40(%rsp), %r12                                #[spill]
L_LCFI45:
        movq      32(%rsp), %r13                                #[spill]
L_LCFI46:
        movq      24(%rsp), %r14                                #[spill]
L_LCFI47:
        movq      16(%rsp), %r15                                #[spill]
L_LCFI48:
        movq      8(%rsp), %rbx                                 #[spill]
L_LCFI49:
        movq      (%rsp), %rbp                                  #[spill]
L_LCFI50:
                                # LOE rbx rbp r12 r13 r14 r15
L_B4.10:                        # Preds L_B4.9 L_B4.1
                                # Execution count [1.00e+00]
        lea       il0_peep_printf_format_1(%rip), %rdi          #75.2
        addq      $56, %rsp                                     #75.2
L_LCFI51:
        jmp       _puts                                         #75.2
        .align    4
                                # LOE
L_LCFI52:
# mark_end;
L..LN_Print2D.3:
	.cstring
	.space 1	# pad
	.align 2
il0_peep_printf_format_1:
	.long	2021161080
	.word	30840
	.byte	0
	.section	__DATA, __data
# -- End  _Print2D
	.section	__TEXT, __text
L_2__routine_start__Initialize2D_4:
# -- Begin  _Initialize2D
	.section	__TEXT, __text
# mark_begin;
       .align    4
	.globl _Initialize2D
# --- Initialize2D(complex128_t **, const int *)
_Initialize2D:
# parameter 1: %rdi
# parameter 2: %rsi
L_B5.1:                         # Preds L_B5.0
                                # Execution count [1.00e+00]
L_LCFI53:
L____tag_value__Initialize2D.122:
L_L123:
                                                        #27.58
        subq      $24, %rsp                                     #27.58
L_LCFI54:
        movq      %rdi, %rdx                                    #27.58
        cmpl      $0, (%rsi)                                    #31.18
        jle       L_B5.13       # Prob 10%                      #31.18
                                # LOE rdx rbx rbp rsi r12 r13 r14 r15
L_B5.2:                         # Preds L_B5.1
                                # Execution count [9.00e-01]
        xorl      %eax, %eax                                    #31.2
        movl      4(%rsi), %edi                                 #32.42
        xorps     %xmm0, %xmm0                                  #34.16
        movq      %r12, 16(%rsp)                                #34.16[spill]
L_LCFI55:
        movq      %rsi, %r12                                    #34.16
        movq      %r13, 8(%rsp)                                 #34.16[spill]
L_LCFI56:
        movq      %rdx, %r13                                    #34.16
        movq      %r14, (%rsp)                                  #34.16[spill]
L_LCFI57:
        movq      %rax, %r14                                    #34.16
                                # LOE rbx rbp r12 r13 r14 r15 edi
L_B5.3:                         # Preds L_B5.11 L_B5.2
                                # Execution count [5.00e+00]
        movslq    %edi, %rdi                                    #32.12
        incq      %r14                                          #31.2
        shlq      $4, %rdi                                      #32.12
#       malloc(size_t)
        call      _malloc                                       #32.12
                                # LOE rax rbx rbp r12 r13 r14 r15
L_B5.4:                         # Preds L_B5.3
                                # Execution count [5.00e+00]
        movl      4(%r12), %edi                                 #33.19
        testl     %edi, %edi                                    #33.19
        movq      %rax, -8(%r13,%r14,8)                         #32.3
        jle       L_B5.11       # Prob 50%                      #33.19
                                # LOE rax rbx rbp r12 r13 r14 r15 edi
L_B5.5:                         # Preds L_B5.4
                                # Execution count [5.00e+00]
        movl      %edi, %ecx                                    #33.3
        xorl      %r9d, %r9d                                    #33.3
        shrl      $1, %ecx                                      #33.3
        movl      $1, %r10d                                     #33.3
        xorl      %r8d, %r8d                                    #33.3
        testl     %ecx, %ecx                                    #33.3
        jbe       L_B5.9        # Prob 10%                      #33.3
                                # LOE rax rcx rbx rbp r8 r9 r12 r13 r14 r15 edi r10d
L_B5.6:                         # Preds L_B5.5
                                # Execution count [4.50e+00]
        xorps     %xmm0, %xmm0                                  #33.3
                                # LOE rax rcx rbx rbp r8 r9 r12 r13 r14 r15 edi xmm0
L_B5.7:                         # Preds L_B5.7 L_B5.6
                                # Execution count [1.25e+01]
        incq      %r9                                           #33.3
        movsd     %xmm0, (%r8,%rax)                             #34.4
        movhpd    %xmm0, 8(%r8,%rax)                            #34.4
        movsd     %xmm0, 16(%r8,%rax)                           #34.4
        movhpd    %xmm0, 24(%r8,%rax)                           #34.4
        addq      $32, %r8                                      #33.3
        cmpq      %rcx, %r9                                     #33.3
        jb        L_B5.7        # Prob 64%                      #33.3
                                # LOE rax rcx rbx rbp r8 r9 r12 r13 r14 r15 edi xmm0
L_B5.8:                         # Preds L_B5.7
                                # Execution count [4.50e+00]
        lea       1(%r9,%r9), %r10d                             #34.4
                                # LOE rax rbx rbp r12 r13 r14 r15 edi r10d
L_B5.9:                         # Preds L_B5.8 L_B5.5
                                # Execution count [5.00e+00]
        lea       -1(%r10), %ecx                                #33.3
        cmpl      %edi, %ecx                                    #33.3
        jae       L_B5.11       # Prob 10%                      #33.3
                                # LOE rax rbx rbp r12 r13 r14 r15 edi r10d
L_B5.10:                        # Preds L_B5.9
                                # Execution count [4.50e+00]
        movslq    %r10d, %r10                                   #33.3
        shlq      $4, %r10                                      #33.3
        xorps     %xmm0, %xmm0                                  #34.4
        movsd     %xmm0, -16(%r10,%rax)                         #34.4
        movhpd    %xmm0, -8(%r10,%rax)                          #34.4
                                # LOE rbx rbp r12 r13 r14 r15 edi
L_B5.11:                        # Preds L_B5.9 L_B5.4 L_B5.10
                                # Execution count [5.00e+00]
        movslq    (%r12), %rcx                                  #31.18
        cmpq      %rcx, %r14                                    #31.18
        jl        L_B5.3        # Prob 82%                      #31.18
                                # LOE rbx rbp r12 r13 r14 r15 edi
L_B5.12:                        # Preds L_B5.11
                                # Execution count [9.00e-01]
        movq      16(%rsp), %r12                                #[spill]
L_LCFI58:
        movq      8(%rsp), %r13                                 #[spill]
L_LCFI59:
        movq      (%rsp), %r14                                  #[spill]
L_LCFI60:
                                # LOE rbx rbp r12 r13 r14 r15
L_B5.13:                        # Preds L_B5.12 L_B5.1
                                # Execution count [1.00e+00]
        addq      $24, %rsp                                     #36.1
L_LCFI61:
        ret                                                     #36.1
        .align    4
                                # LOE
L_LCFI62:
# mark_end;
L..LN_Initialize2D.4:
	.section	__DATA, __data
# -- End  _Initialize2D
	.section	__TEXT, __text
L_2__routine_start__Initialize4D_5:
# -- Begin  _Initialize4D
	.section	__TEXT, __text
# mark_begin;
       .align    4
	.globl _Initialize4D
# --- Initialize4D(complex128_t ****, const int *)
_Initialize4D:
# parameter 1: %rdi
# parameter 2: %rsi
L_B6.1:                         # Preds L_B6.0
                                # Execution count [1.00e+00]
L_LCFI63:
L____tag_value__Initialize4D.138:
L_L139:
                                                        #9.60
        subq      $88, %rsp                                     #9.60
L_LCFI64:
        movq      %rdi, %rdx                                    #9.60
        xorl      %eax, %eax                                    #14.7
        cmpl      $0, (%rsi)                                    #14.18
        jle       L_B6.22       # Prob 10%                      #14.18
                                # LOE rax rdx rbx rbp rsi r12 r13 r14 r15
L_B6.2:                         # Preds L_B6.1
                                # Execution count [9.00e-01]
        movslq    4(%rsi), %rdi                                 #15.45
        movq      %r12, (%rsp)                                  #15.45[spill]
L_LCFI65:
        movq      %rax, %r12                                    #15.45
        movq      %r13, 8(%rsp)                                 #15.45[spill]
        movq      %r14, 16(%rsp)                                #15.45[spill]
L_LCFI66:
        movq      %rdx, %r14                                    #15.45
        movq      %r15, 24(%rsp)                                #15.45[spill]
L_LCFI67:
        movq      %rsi, %r15                                    #15.45
        movq      %rbx, 32(%rsp)                                #15.45[spill]
        movq      %rbp, 40(%rsp)                                #15.45[spill]
L_LCFI68:
                                # LOE rdi r12 r14 r15
L_B6.3:                         # Preds L_B6.20 L_B6.2
                                # Execution count [5.00e+00]
        shlq      $3, %rdi                                      #15.12
#       malloc(size_t)
        call      _malloc                                       #15.12
                                # LOE rax r12 r14 r15
L_B6.4:                         # Preds L_B6.3
                                # Execution count [5.00e+00]
        movslq    4(%r15), %rdi                                 #16.19
        xorl      %r13d, %r13d                                  #16.8
        movq      %rax, (%r14,%r12,8)                           #15.3
        testq     %rdi, %rdi                                    #16.19
        jle       L_B6.20       # Prob 10%                      #16.19
                                # LOE rdi r12 r13 r14 r15
L_B6.5:                         # Preds L_B6.4
                                # Execution count [4.50e+00]
        movslq    8(%r15), %rcx                                 #17.48
                                # LOE rcx r12 r13 r14 r15
L_B6.6:                         # Preds L_B6.18 L_B6.5
                                # Execution count [2.50e+01]
        shlq      $3, %rcx                                      #17.16
        movq      %rcx, %rdi                                    #17.16
#       malloc(size_t)
        call      _malloc                                       #17.16
                                # LOE rax r12 r13 r14 r15
L_B6.7:                         # Preds L_B6.6
                                # Execution count [2.50e+01]
        movq      (%r14,%r12,8), %rcx                           #17.4
        xorl      %ebx, %ebx                                    #18.9
        movq      %rax, (%rcx,%r13,8)                           #17.4
        movslq    8(%r15), %rcx                                 #18.20
        testq     %rcx, %rcx                                    #18.20
        jle       L_B6.18       # Prob 10%                      #18.20
                                # LOE rcx rbx r12 r13 r14 r15
L_B6.8:                         # Preds L_B6.7
                                # Execution count [2.25e+01]
        movslq    12(%r15), %rbp                                #19.50
                                # LOE rbx rbp r12 r13 r14 r15
L_B6.9:                         # Preds L_B6.16 L_B6.8
                                # Execution count [1.25e+02]
        shlq      $4, %rbp                                      #19.20
        movq      %rbp, %rdi                                    #19.20
#       malloc(size_t)
        call      _malloc                                       #19.20
                                # LOE rax rbx r12 r13 r14 r15
L_B6.10:                        # Preds L_B6.9
                                # Execution count [1.25e+02]
        movq      (%r14,%r12,8), %rcx                           #19.5
        movq      (%rcx,%r13,8), %rbp                           #19.5
        xorl      %ecx, %ecx                                    #20.10
        xorl      %edx, %edx                                    #20.10
        movq      %rax, (%rbp,%rbx,8)                           #19.5
        movslq    12(%r15), %rbp                                #20.21
        testq     %rbp, %rbp                                    #20.21
        jle       L_B6.16       # Prob 10%                      #20.21
                                # LOE rdx rcx rbx rbp r12 r13 r14 r15
L_B6.11:                        # Preds L_B6.10
                                # Execution count [1.12e+02]
        movq      (%r14,%r12,8), %rax                           #21.6
        movq      %r12, 64(%rsp)                                #21.6[spill]
        movq      %rax, %r12                                    #21.6
        movq      %r14, 56(%rsp)                                #21.6[spill]
        movq      %rdx, %r14                                    #21.6
        movq      %r15, 48(%rsp)                                #21.6[spill]
        movq      %rcx, %r15                                    #21.6
                                # LOE rbx rbp r12 r13 r14 r15
L_B6.12:                        # Preds L_B6.31 L_B6.11
                                # Execution count [6.25e+02]
#       rand(void)
        call      _rand                                         #21.32
                                # LOE rbx rbp r12 r13 r14 r15 eax
L_B6.28:                        # Preds L_B6.12
                                # Execution count [6.25e+02]
        movl      %eax, %r9d                                    #21.32
                                # LOE rbx rbp r12 r13 r14 r15 r9d
L_B6.13:                        # Preds L_B6.28
                                # Execution count [6.25e+02]
        movl      $1374389535, %eax                             #21.24
        movl      %r9d, %esi                                    #21.24
        imull     %r9d                                          #21.24
        sarl      $31, %esi                                     #21.24
        xorps     %xmm1, %xmm1                                  #21.24
        sarl      $5, %edx                                      #21.24
        subl      %esi, %edx                                    #21.24
        imull     $-100, %edx, %r8d                             #21.24
        movsd     L_2il0floatpacket.0(%rip), %xmm0              #21.24
        addl      %r8d, %r9d                                    #21.24
        cvtsi2sd  %r9d, %xmm1                                   #21.24
        call      _pow                                          #21.24
                                # LOE rbx rbp r12 r13 r14 r15 xmm0
L_B6.30:                        # Preds L_B6.13
                                # Execution count [6.25e+02]
        movsd     %xmm0, 72(%rsp)                               #21.24[spill]
#       rand(void)
        call      _rand                                         #21.60
                                # LOE rbx rbp r12 r13 r14 r15 eax
L_B6.29:                        # Preds L_B6.30
                                # Execution count [6.25e+02]
        movl      %eax, %r9d                                    #21.60
                                # LOE rbx rbp r12 r13 r14 r15 r9d
L_B6.14:                        # Preds L_B6.29
                                # Execution count [6.25e+02]
        movl      $1374389535, %eax                             #21.52
        movl      %r9d, %esi                                    #21.52
        imull     %r9d                                          #21.52
        sarl      $31, %esi                                     #21.52
        xorps     %xmm1, %xmm1                                  #21.52
        sarl      $5, %edx                                      #21.52
        subl      %esi, %edx                                    #21.52
        imull     $-100, %edx, %r8d                             #21.52
        movsd     L_2il0floatpacket.0(%rip), %xmm0              #21.52
        addl      %r8d, %r9d                                    #21.52
        cvtsi2sd  %r9d, %xmm1                                   #21.52
        call      _pow                                          #21.52
                                # LOE rbx rbp r12 r13 r14 r15 xmm0
L_B6.31:                        # Preds L_B6.14
                                # Execution count [6.25e+02]
        movq      (%r12,%r13,8), %rsi                           #21.6
        movaps    %xmm0, %xmm1                                  #21.52
        movsd     72(%rsp), %xmm0                               #21.6[spill]
        incq      %r15                                          #20.31
        movq      (%rsi,%rbx,8), %r8                            #21.6
        movsd     %xmm0, (%r8,%r14)                             #21.6
        movsd     %xmm1, 8(%r8,%r14)                            #21.6
        addq      $16, %r14                                     #20.31
        cmpq      %rbp, %r15                                    #20.21
        jl        L_B6.12       # Prob 82%                      #20.21
                                # LOE rbx rbp r12 r13 r14 r15
L_B6.15:                        # Preds L_B6.31
                                # Execution count [1.13e+02]
        movq      64(%rsp), %r12                                #[spill]
        movq      56(%rsp), %r14                                #[spill]
        movq      48(%rsp), %r15                                #[spill]
                                # LOE rbx rbp r12 r13 r14 r15
L_B6.16:                        # Preds L_B6.15 L_B6.10
                                # Execution count [1.25e+02]
        incq      %rbx                                          #18.30
        movslq    8(%r15), %rcx                                 #18.20
        cmpq      %rcx, %rbx                                    #18.20
        jl        L_B6.9        # Prob 82%                      #18.20
                                # LOE rcx rbx rbp r12 r13 r14 r15
L_B6.18:                        # Preds L_B6.16 L_B6.7
                                # Execution count [2.50e+01]
        .byte     15                                            #16.29
        .byte     31                                            #16.29
        .byte     64                                            #16.29
        .byte     0                                             #16.29
        incq      %r13                                          #16.29
        movslq    4(%r15), %rdi                                 #16.19
        cmpq      %rdi, %r13                                    #16.19
        jl        L_B6.6        # Prob 82%                      #16.19
                                # LOE rcx rdi r12 r13 r14 r15
L_B6.20:                        # Preds L_B6.18 L_B6.4
                                # Execution count [5.00e+00]
        .byte     15                                            #14.28
        .byte     31                                            #14.28
        .byte     68                                            #14.28
        .byte     0                                             #14.28
        .byte     0                                             #14.28
        incq      %r12                                          #14.28
        movslq    (%r15), %rcx                                  #14.18
        cmpq      %rcx, %r12                                    #14.18
        jl        L_B6.3        # Prob 82%                      #14.18
                                # LOE rdi r12 r14 r15
L_B6.21:                        # Preds L_B6.20
                                # Execution count [9.00e-01]
        movq      (%rsp), %r12                                  #[spill]
L_LCFI69:
        movq      8(%rsp), %r13                                 #[spill]
L_LCFI70:
        movq      16(%rsp), %r14                                #[spill]
L_LCFI71:
        movq      24(%rsp), %r15                                #[spill]
L_LCFI72:
        movq      32(%rsp), %rbx                                #[spill]
L_LCFI73:
        movq      40(%rsp), %rbp                                #[spill]
L_LCFI74:
                                # LOE rbx rbp r12 r13 r14 r15
L_B6.22:                        # Preds L_B6.21 L_B6.1
                                # Execution count [1.00e+00]
        addq      $88, %rsp                                     #25.1
L_LCFI75:
        ret                                                     #25.1
        .align    4
                                # LOE
L_LCFI76:
# mark_end;
L..LN_Initialize4D.5:
	.section	__DATA, __data
# -- End  _Initialize4D
	.cstring
	.space 1	# pad
	.align 2
L_2__STRING.0:
	.long	1534227041
	.long	1532847141
	.long	1532847141
	.long	1532847141
	.long	542991397
	.long	1730486333
	.long	1763715872
	.long	174531872
	.byte	0
	.space 3	# pad
	.align 2
L_2__STRING.2:
	.long	1680169795
	.long	1680169821
	.long	540876893
	.long	723543845
	.long	622881056
	.word	2663
	.byte	0
	.literal8
	.align 3
	.align 3
L_2il0floatpacket.0:
	.long	0x00000000,0xbff00000
	.section	__DATA, __data
	.globl _Free4D.eh
	.globl _Free2D.eh
	.globl _Print4D.eh
	.globl _Print2D.eh
	.globl _Initialize2D.eh
	.globl _Initialize4D.eh
// -- Begin SEGMENT __eh_frame
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
__eh_frame_seg:
L.__eh_frame_seg:
EH_frame0:
L_fde_cie_0:
	.long 0x00000014
	.long 0x00000000
	.long 0x00527a01
	.long 0x01107801
	.long 0x08070c10
	.long 0x01900190
_Free4D.eh:
	.long 0x0000006c
	.long 0x0000001c
	.quad L_LCFI1-_Free4D.eh-0x8
	.set L_Qlab1,L_LCFI15-L_LCFI1
	.quad L_Qlab1
	.short 0x0400
	.set L_lab1,L_LCFI2-L_LCFI1
	.long L_lab1
	.short 0x600e
	.byte 0x04
	.set L_lab2,L_LCFI3-L_LCFI2
	.long L_lab2
	.short 0x0c8c
	.byte 0x04
	.set L_lab3,L_LCFI4-L_LCFI3
	.long L_lab3
	.short 0x0b8d
	.byte 0x04
	.set L_lab4,L_LCFI5-L_LCFI4
	.long L_lab4
	.short 0x0a8e
	.byte 0x04
	.set L_lab5,L_LCFI6-L_LCFI5
	.long L_lab5
	.long 0x098f0883
	.byte 0x04
	.set L_lab6,L_LCFI7-L_LCFI6
	.long L_lab6
	.short 0x0786
	.byte 0x04
	.set L_lab7,L_LCFI8-L_LCFI7
	.long L_lab7
	.short 0x04cc
	.set L_lab8,L_LCFI9-L_LCFI8
	.long L_lab8
	.short 0x04cd
	.set L_lab9,L_LCFI10-L_LCFI9
	.long L_lab9
	.short 0x04ce
	.set L_lab10,L_LCFI11-L_LCFI10
	.long L_lab10
	.short 0x04cf
	.set L_lab11,L_LCFI12-L_LCFI11
	.long L_lab11
	.short 0x04c3
	.set L_lab12,L_LCFI13-L_LCFI12
	.long L_lab12
	.short 0x04c6
	.set L_lab13,L_LCFI14-L_LCFI13
	.long L_lab13
	.short 0x080e
_Free2D.eh:
	.long 0x0000004c
	.long 0x0000008c
	.quad L_LCFI16-_Free2D.eh-0x8
	.set L_Qlab2,L_LCFI25-L_LCFI16
	.quad L_Qlab2
	.short 0x0400
	.set L_lab14,L_LCFI17-L_LCFI16
	.long L_lab14
	.short 0x200e
	.byte 0x04
	.set L_lab15,L_LCFI18-L_LCFI17
	.long L_lab15
	.short 0x028c
	.byte 0x04
	.set L_lab16,L_LCFI19-L_LCFI18
	.long L_lab16
	.short 0x038d
	.byte 0x04
	.set L_lab17,L_LCFI20-L_LCFI19
	.long L_lab17
	.short 0x048e
	.byte 0x04
	.set L_lab18,L_LCFI21-L_LCFI20
	.long L_lab18
	.short 0x04cc
	.set L_lab19,L_LCFI22-L_LCFI21
	.long L_lab19
	.short 0x04cd
	.set L_lab20,L_LCFI23-L_LCFI22
	.long L_lab20
	.short 0x04ce
	.set L_lab21,L_LCFI24-L_LCFI23
	.long L_lab21
	.long 0x0000080e
_Print4D.eh:
	.long 0x0000006c
	.long 0x000000dc
	.quad L_LCFI26-_Print4D.eh-0x8
	.set L_Qlab3,L_LCFI39-L_LCFI26
	.quad L_Qlab3
	.short 0x0400
	.set L_lab22,L_LCFI27-L_LCFI26
	.long L_lab22
	.short 0x600e
	.byte 0x04
	.set L_lab23,L_LCFI28-L_LCFI27
	.long L_lab23
	.short 0x078c
	.byte 0x04
	.set L_lab24,L_LCFI29-L_LCFI28
	.long L_lab24
	.short 0x088d
	.byte 0x04
	.set L_lab25,L_LCFI30-L_LCFI29
	.long L_lab25
	.long 0x098e0b83
	.short 0x0a8f
	.byte 0x04
	.set L_lab26,L_LCFI31-L_LCFI30
	.long L_lab26
	.short 0x0c86
	.byte 0x04
	.set L_lab27,L_LCFI32-L_LCFI31
	.long L_lab27
	.short 0x04cc
	.set L_lab28,L_LCFI33-L_LCFI32
	.long L_lab28
	.short 0x04cd
	.set L_lab29,L_LCFI34-L_LCFI33
	.long L_lab29
	.short 0x04ce
	.set L_lab30,L_LCFI35-L_LCFI34
	.long L_lab30
	.short 0x04cf
	.set L_lab31,L_LCFI36-L_LCFI35
	.long L_lab31
	.short 0x04c3
	.set L_lab32,L_LCFI37-L_LCFI36
	.long L_lab32
	.short 0x04c6
	.set L_lab33,L_LCFI38-L_LCFI37
	.long L_lab33
	.long 0x0000080e
	.short 0x0000
	.byte 0x00
_Print2D.eh:
	.long 0x00000064
	.long 0x0000014c
	.quad L_LCFI40-_Print2D.eh-0x8
	.set L_Qlab4,L_LCFI52-L_LCFI40
	.quad L_Qlab4
	.short 0x0400
	.set L_lab34,L_LCFI41-L_LCFI40
	.long L_lab34
	.short 0x400e
	.byte 0x04
	.set L_lab35,L_LCFI42-L_LCFI41
	.long L_lab35
	.long 0x048d038c
	.byte 0x04
	.set L_lab36,L_LCFI43-L_LCFI42
	.long L_lab36
	.short 0x058e
	.byte 0x04
	.set L_lab37,L_LCFI44-L_LCFI43
	.long L_lab37
	.long 0x08860783
	.short 0x068f
	.byte 0x04
	.set L_lab38,L_LCFI45-L_LCFI44
	.long L_lab38
	.short 0x04cc
	.set L_lab39,L_LCFI46-L_LCFI45
	.long L_lab39
	.short 0x04cd
	.set L_lab40,L_LCFI47-L_LCFI46
	.long L_lab40
	.short 0x04ce
	.set L_lab41,L_LCFI48-L_LCFI47
	.long L_lab41
	.short 0x04cf
	.set L_lab42,L_LCFI49-L_LCFI48
	.long L_lab42
	.short 0x04c3
	.set L_lab43,L_LCFI50-L_LCFI49
	.long L_lab43
	.short 0x04c6
	.set L_lab44,L_LCFI51-L_LCFI50
	.long L_lab44
	.long 0x0000080e
_Initialize2D.eh:
	.long 0x0000004c
	.long 0x000001b4
	.quad L_LCFI53-_Initialize2D.eh-0x8
	.set L_Qlab5,L_LCFI62-L_LCFI53
	.quad L_Qlab5
	.short 0x0400
	.set L_lab45,L_LCFI54-L_LCFI53
	.long L_lab45
	.short 0x200e
	.byte 0x04
	.set L_lab46,L_LCFI55-L_LCFI54
	.long L_lab46
	.short 0x028c
	.byte 0x04
	.set L_lab47,L_LCFI56-L_LCFI55
	.long L_lab47
	.short 0x038d
	.byte 0x04
	.set L_lab48,L_LCFI57-L_LCFI56
	.long L_lab48
	.short 0x048e
	.byte 0x04
	.set L_lab49,L_LCFI58-L_LCFI57
	.long L_lab49
	.short 0x04cc
	.set L_lab50,L_LCFI59-L_LCFI58
	.long L_lab50
	.short 0x04cd
	.set L_lab51,L_LCFI60-L_LCFI59
	.long L_lab51
	.short 0x04ce
	.set L_lab52,L_LCFI61-L_LCFI60
	.long L_lab52
	.long 0x0000080e
_Initialize4D.eh:
	.long 0x0000006c
	.long 0x00000204
	.quad L_LCFI63-_Initialize4D.eh-0x8
	.set L_Qlab6,L_LCFI76-L_LCFI63
	.quad L_Qlab6
	.short 0x0400
	.set L_lab53,L_LCFI64-L_LCFI63
	.long L_lab53
	.short 0x600e
	.byte 0x04
	.set L_lab54,L_LCFI65-L_LCFI64
	.long L_lab54
	.short 0x0c8c
	.byte 0x04
	.set L_lab55,L_LCFI66-L_LCFI65
	.long L_lab55
	.long 0x0a8e0b8d
	.byte 0x04
	.set L_lab56,L_LCFI67-L_LCFI66
	.long L_lab56
	.short 0x098f
	.byte 0x04
	.set L_lab57,L_LCFI68-L_LCFI67
	.long L_lab57
	.long 0x07860883
	.byte 0x04
	.set L_lab58,L_LCFI69-L_LCFI68
	.long L_lab58
	.short 0x04cc
	.set L_lab59,L_LCFI70-L_LCFI69
	.long L_lab59
	.short 0x04cd
	.set L_lab60,L_LCFI71-L_LCFI70
	.long L_lab60
	.short 0x04ce
	.set L_lab61,L_LCFI72-L_LCFI71
	.long L_lab61
	.short 0x04cf
	.set L_lab62,L_LCFI73-L_LCFI72
	.long L_lab62
	.short 0x04c3
	.set L_lab63,L_LCFI74-L_LCFI73
	.long L_lab63
	.short 0x04c6
	.set L_lab64,L_LCFI75-L_LCFI74
	.long L_lab64
	.long 0x0000080e
	.short 0x0000
	.byte 0x00
# End
	.subsections_via_symbols
