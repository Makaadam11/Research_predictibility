'use client'

import { useState, useEffect } from 'react'
import SingleSelect from './Questionaire/SingleSelect'
import Slider from './Questionaire/Slider'
import { Question, QuestionarieData, getQuestions, QuestionaireProps } from '../types/QuestionaireTypes'
import MultiSelect from './Questionaire/MultiSelect'
import DropdownSelect from './Questionaire/DropdownSelect'
import { TextInput } from './Questionaire/TextInput'
import { submitQuestionaire } from '../api/data'
import { loadCourses } from '../api/data'

export default function Questionaire({ 
  university, 
  formData, 
  setFormData, 
  onSubmitSuccess 
}: QuestionaireProps) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false); // New state for submission status
  const [error, setError] = useState<string | null>(null);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadQuestions = async () => {
      try {
        const courses = await loadCourses(university);
        const loadedQuestions = await getQuestions(university, courses);
        setQuestions(loadedQuestions);
        setIsLoading(false);
      } catch (err) {
        setError('Failed to load questions');
        setIsLoading(false);
      }
    };

    loadQuestions();
  }, [university]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setIsSubmitted(true); // Set submitted state immediately
    setError(null);
  
    const answers = questions.map(question => ({
      id: question.id,
      answer: formData[question.id]
    }));
  
    submitQuestionaire({
      answers,
      source: university.toUpperCase()
    })
      .then(() => {
        onSubmitSuccess();
        setFormData({}); // Clear formData after successful submission
      })
      .catch(() => {
        setError('Failed to submit survey');
        setIsSubmitted(false); // Reset submitted state on error
      })
      .finally(() => {
        setIsSubmitting(false);
      });
  };

  if (isLoading) return <div>Loading questions...</div>;
  if (error) return <div>Error: {error}</div>;

  const renderQuestion = (question: Question) => {
    switch (question.type) {
      case 'single':
        return (
          <div key={question.id}>
            <SingleSelect
              question={question.question}
              options={question.options || []}
              value={formData[question.id] || ''}
              onValueChange={(value) => setFormData((prev: any) => ({...prev, [question.id]: value}))}
            />
          </div>
        )
      case 'slider':
        return (
          <div key={question.id}>
            <Slider
              question={question.question}
              min={question.min || 0}
              max={question.max || 100}
              step={question.step || 1}
              value={formData[question.id] || 0}
              showAboveMax={question.showAboveMax}
              onValueChange={(value) => setFormData((prev: any) => ({...prev, [question.id]: value}))}
            />
          </div>
        )
      case 'multi':
        return (
          <div key={question.id}>
            <MultiSelect
              question={question.question}
              options={question.options || []}
              value={formData[question.id] || []}
              onValueChange={(value) => setFormData((prev: any) => ({...prev, [question.id]: value}))}
            />
          </div>
        )
      case 'text':
        return (
          <div key={question.id}>
            <TextInput
              question={question.question}
              value={formData[question.id] || ''}
              onValueChange={(value) => setFormData((prev: any) => ({...prev, [question.id]: value}))}
            />
          </div>
        )
      case 'dropdown':
        return (
          <div key={question.id}>
            <DropdownSelect
              question={question.question}
              options={question.options || []}
              value={formData[question.id] || ''}
              onValueChange={(value) => setFormData((prev: any) => ({...prev, [question.id]: value}))}
            />
          </div>
        )
      case 'textinput':
        return (
          <div key={question.id}>
            <TextInput
              question={question.question}
              value={formData[question.id] || ''}
              onValueChange={(value) => setFormData((prev: any) => ({...prev, [question.id]: value}))}
            />
          </div>
        )
      default:
        return null;
    }
  }

  return (
    <div>
      <form onSubmit={handleSubmit} className="space-y-8 p-6 bg-white rounded-lg shadow">
        <h1 className="text-2xl font-bold mb-6">Student Survey</h1>
        {error && (
          <div className="p-4 bg-red-100 text-red-700 rounded mb-4">
            {error}
          </div>
        )}
        {isSubmitted && !error && (
          <div className="p-4 bg-green-100 text-green-700 rounded mb-4">
            Submitted successfully!
          </div>
        )}
        {questions.map(renderQuestion)}
        <button 
          type="submit" 
          disabled={isSubmitting}
          className={`w-full py-2 px-4 ${isSubmitting ? 'bg-blue-400' : 'bg-blue-600'} text-white rounded hover:bg-blue-700 transition-colors`}
        >
          {isSubmitting ? 'Submitted' : 'Submit'}
        </button>
      </form>
    </div>
  );
};