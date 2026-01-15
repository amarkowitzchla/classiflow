import { useState } from 'react';
import { MessageSquare, Send, Trash2, User } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useComments, useCreateComment, useDeleteComment } from '../hooks/useApi';
import type { Comment } from '../types/api';

interface CommentsProps {
  scopeType: 'project' | 'run' | 'artifact';
  scopeId: string;
}

export function Comments({ scopeType, scopeId }: CommentsProps) {
  const { data, isLoading, error } = useComments(scopeType, scopeId);
  const createMutation = useCreateComment();
  const deleteMutation = useDeleteComment();

  const [newComment, setNewComment] = useState('');
  const [author, setAuthor] = useState(() => localStorage.getItem('comment_author') || '');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newComment.trim() || !author.trim()) return;

    // Save author for next time
    localStorage.setItem('comment_author', author);

    await createMutation.mutateAsync({
      scope_type: scopeType,
      scope_id: scopeId,
      author: author.trim(),
      content: newComment.trim(),
    });

    setNewComment('');
  };

  const handleDelete = async (commentId: number) => {
    if (confirm('Delete this comment?')) {
      await deleteMutation.mutateAsync(commentId);
    }
  };

  if (isLoading) {
    return <div className="text-gray-500 text-sm">Loading comments...</div>;
  }

  if (error) {
    return <div className="text-red-500 text-sm">Failed to load comments</div>;
  }

  const comments = data?.items || [];

  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-2 text-sm font-medium text-gray-700">
        <MessageSquare className="w-4 h-4" />
        <span>Comments ({data?.total || 0})</span>
      </div>

      {/* Comment list */}
      <div className="space-y-3">
        {comments.map((comment) => (
          <CommentCard
            key={comment.id}
            comment={comment}
            onDelete={() => handleDelete(comment.id)}
          />
        ))}
        {comments.length === 0 && (
          <p className="text-gray-400 text-sm">No comments yet</p>
        )}
      </div>

      {/* New comment form */}
      <form onSubmit={handleSubmit} className="space-y-3 pt-4 border-t border-gray-100">
        <div className="flex items-center space-x-2">
          <User className="w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
            placeholder="Your name"
            className="flex-1 text-sm border border-gray-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
        <div className="flex items-start space-x-2">
          <textarea
            value={newComment}
            onChange={(e) => setNewComment(e.target.value)}
            placeholder="Add a comment..."
            rows={2}
            className="flex-1 text-sm border border-gray-200 rounded px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none"
          />
          <button
            type="submit"
            disabled={!newComment.trim() || !author.trim() || createMutation.isPending}
            className="p-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </form>
    </div>
  );
}

interface CommentCardProps {
  comment: Comment;
  onDelete: () => void;
}

function CommentCard({ comment, onDelete }: CommentCardProps) {
  return (
    <div className="bg-gray-50 rounded-lg p-3 group">
      <div className="flex items-start justify-between">
        <div className="flex items-center space-x-2 text-sm">
          <span className="font-medium text-gray-900">{comment.author}</span>
          <span className="text-gray-400">
            {formatDistanceToNow(new Date(comment.created_at), { addSuffix: true })}
          </span>
        </div>
        <button
          onClick={onDelete}
          className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 transition-opacity"
        >
          <Trash2 className="w-3 h-3" />
        </button>
      </div>
      <p className="mt-1 text-sm text-gray-700 whitespace-pre-wrap">{comment.content}</p>
    </div>
  );
}
